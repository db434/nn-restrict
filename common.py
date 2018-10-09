import os
import sys

import torch
import torch.backends.cudnn as cudnn

import models
import modifiers
import structured
import training
import util


def process(dataset):
    """
    Wrap the main task with some exception handlers. Otherwise a huge,
    unnecessary, parallel stacktrace is printed.

    :param dataset: an object from `datasets` describing the data to be used.
    """
    try:
        _process(dataset)
    except KeyboardInterrupt:
        # Note that due to a Python multiprocessing issue, the stack trace isn't
        # actually prevented here. https://discuss.pytorch.org/t/9740
        util.log.error("Process terminated by user.")
        sys.exit()


def _process(dataset):
    """
    Initialise the model according to the command line arguments, and use it
    as the user requests.

    :param dataset: an object from `datasets` describing the data to be used.
    """
    best_top1 = 0
    
    args = util.args.parse_args(dataset)

    conv_type = structured.conv2d_types[args.conv_type]
    unique_name = create_unique_name(args)

    # Create model
    model = models.get_model(args.arch, distributed=args.distributed,
                             use_cuda=args.cuda,
                             input_channels=dataset.input_channels(),
                             num_classes=dataset.num_classes(),
                             conv2d=conv_type,
                             args=args)

    # Restrict the datatypes used within the model, if necessary.
    model = restrict_datatypes(model, args)

    schedule = get_lr_schedule(dataset, args)

    if args.distill:
        trainer = training.distillation.Trainer(dataset, model, schedule, args)
    else:
        trainer = training.trainer.Trainer(dataset, model, schedule, args)

    # Record whether this model is being generated from scratch.
    new_model = True

    # Resume from a checkpoint
    if args.undump_dir:
        # Load from numpy arrays. This does not load any optimiser state,
        # so does not support training.
        util.stats.data_restore(args.undump_dir, model)
        new_model = False
    elif args.resume or args.evaluate or args.generate:
        # Proper load from a checkpoint
        if args.model_file:
            start_epoch, best_top1 = \
                util.checkpoint.load_path(args.model_file, model,
                                          trainer.optimiser)
        else:
            start_epoch, best_top1 = \
                util.checkpoint.load(args.save_dir, unique_name, model,
                                     trainer.optimiser)
        args.start_epoch = start_epoch
        new_model = False

    cudnn.benchmark = True

    # Quick analysis before training.
    if args.evaluate:
        trainer.validate()
    
    if args.stats:
        analyse(trainer, args)
        print(util.stats.computation_cost_csv(unique_name))

    if args.dump_weights or args.dump_acts or args.dump_grads:
        dump_data(trainer, args)
        return
    
    if args.gradients:
        collect_gradients(unique_name, args.save_dir, args.start_epoch, trainer)

    if args.evaluate or args.stats or args.gradients:
        return

    if args.generate:
        generate_text(model, dataset, args)
        return
        
    # val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
    # util.checkpoint.save(args.save_dir, model, unique_name, optimizer,
    #                      -1, val_top1, True)

    # Store a checkpoint at epoch 0.
    if new_model:
        util.checkpoint.save(args.save_dir, model, unique_name,
                             trainer.optimiser, -1, 0, True)

    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_top1, train_top5 = \
            trainer.train_epoch(epoch)

        # Evaluate on validation set
        val_loss, val_top1, val_top5 = trainer.validate()
        prec1 = val_top1

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_top1
        best_top1 = max(prec1, best_top1)
        util.checkpoint.save(args.save_dir, model, unique_name,
                             trainer.optimiser, epoch, best_top1, is_best)
        
        # Log stats.
        util.checkpoint.log_stats(args.save_dir, unique_name, epoch,
                                  train_loss, train_top1, train_top5,
                                  val_loss, val_top1, val_top5)


def create_unique_name(args):
    """
    Create a unique name for the model, given all the arguments. This name will
    form the default filename.

    :param args: arguments collected by `util.args.parse_args()`
    :return: string name for the model
    """

    conv_type = structured.conv2d_types[args.conv_type]
    unique_name = "{0}-{1}x-{2}".format(args.arch, args.width_multiplier,
                                        args.conv_type)

    if conv_type == structured.butterfly.Conv2d or \
            conv_type == structured.depthwise_butterfly.Conv2d:
        unique_name += "-" + str(args.min_bfly_size)

    if args.grad_noise > 0:
        unique_name += "_gn" + str(args.grad_noise)
    if args.grad_precision > 0:
        unique_name += "_gp" + str(args.grad_precision)
    if args.grad_min > 0:
        unique_name += "_gt" + str(args.grad_min)
    if args.grad_max > 0:
        unique_name += "_gu" + str(args.grad_max)

    if args.act_noise > 0:
        unique_name += "_an" + str(args.act_noise)
    if args.act_precision > 0:
        unique_name += "_ap" + str(args.act_precision)
    if args.act_min > 0:
        unique_name += "_at" + str(args.act_min)
    if args.act_max > 0:
        unique_name += "_au" + str(args.act_max)

    if args.weight_noise > 0:
        unique_name += "_wn" + str(args.weight_noise)
    if args.weight_precision > 0:
        unique_name += "_wp" + str(args.weight_precision)
    if args.weight_min > 0:
        unique_name += "_wt" + str(args.weight_min)
    if args.weight_max > 0:
        unique_name += "_wu" + str(args.weight_max)

    return unique_name


def restrict_datatypes(model, args):
    """
    Apply restrictions to the datatypes used within the model (weights,
    activations and gradients. Restrictions include setting the precision,
    minimum, maximum, and adding noise.

    :param model: `torch.nn.Module` to be restricted.
    :param args: arguments collected by `util.args.parse_args()`
    :return: the original model with restrictions applied
    """
    if args.grad_noise > 0 or args.grad_precision > 0 or args.grad_min > 0 or \
       args.grad_max > 0:
        modifiers.numbers.restrict_gradients(model,
                                             noise=args.grad_noise,
                                             precision=args.grad_precision,
                                             minimum=args.grad_min,
                                             maximum=args.grad_max)

    if args.act_noise > 0 or args.act_precision > 0 or args.act_min > 0 or \
       args.act_max > 0:
        modifiers.numbers.restrict_activations(model,
                                               noise=args.act_noise,
                                               precision=args.act_precision,
                                               minimum=args.act_min,
                                               maximum=args.act_max)

    if args.weight_noise > 0 or args.weight_precision > 0 or \
       args.weight_min > 0 or args.weight_max > 0:
        modifiers.numbers.restrict_weights(model,
                                           noise=args.weight_noise,
                                           precision=args.weight_precision,
                                           minimum=args.weight_min,
                                           maximum=args.weight_max)

    return model


def get_lr_schedule(dataset, args):
    """
    Determine how the learning rate will change as training progresses.

    :param dataset: an object from `datasets` describing the training data.
    :param args: arguments collected by `util.args.parse_args()`
    :return: a `training.lr_schedule.LRSchedule`
    """
    initial_lr = args.lr

    if args.use_restarts:
        period = args.restart_period
        return training.lr_schedule.CosineRestartSchedule(initial_lr, period)
    else:
        steps = dataset.default_lr_steps
        return training.lr_schedule.StepSchedule(initial_lr, steps)


def analyse(trainer, args):
    """
    Train for one batch. Print details about all weights, activations and
    gradients.

    :param trainer: a `training.trainer.Trainer` responsible for training the
                    model
    :param args: arguments collected by `util.args.parse_args()`
    """
    
    # Register hooks on all Modules so they print their details.
    # util.stats.data_distribution_hooks(model, weights=False,
    # activations=False)
    util.stats.computation_cost_hooks(trainer.model)
    train_one_batch(trainer, args)


def dump_data(trainer, args):
    """
    Dump weights, activations and/or gradients for one batch of training.

    :param trainer: a `training.trainer.Trainer` responsible for training the
                    model
    :param args: arguments collected by `util.args.parse_args()`
    """
    assert args.dump_dir is not None
    util.stats.data_dump_hooks(trainer.model, args.dump_dir, args.dump_acts,
                               args.dump_weights, args.dump_grads)
    train_one_batch(trainer, args)


def train_one_batch(trainer, args):
    """
    Train for one minibatch and return.

    :param trainer: a `training.trainer.Trainer` responsible for training the
                    model
    :param args: arguments collected by `util.args.parse_args()`
    """

    # TODO: Accessing a lot of Trainer internals here. Would be nice to
    # encapsulate more.

    # Switch to train mode
    trainer.model.train()

    # I'm not really sure how to access the training data except in a loop,
    # but this doesn't make much sense when we're only using one batch.
    # I'm pretty sure there's a better way than this.
    for data, target in trainer.train_loader:
        if args.cuda:
            target = target.cuda(async=True)

        # Compute output
        output, loss = trainer.minibatch(data, target)

        # Update model
        trainer.optimiser.zero_grad()
        loss.backward()
        trainer.optimiser.step()

        break


def collect_gradients(model_name, directory, epoch, trainer):
    """
    Collect statistics about the gradients seen when training the model and
    write them to a file. Assumes that the model was saved to a file whose
    name includes the current epoch. This does not happen by default.

    :param model_name: string, used to generate file names
    :param directory: directory to find model checkpoint and store results
    :param epoch: number specifying which checkpoint to load
    :param trainer: `training.trainer.Trainer` responsible for training model
    """
                      
    basename = os.path.join(directory, model_name + "_epoch" + str(epoch))
    checkpoint = basename + ".pth.tar"
    log = basename + ".gradients"
    
    # Load this epoch's checkpoint.
    assert os.path.isfile(checkpoint)    
    util.checkpoint.load_path(checkpoint, trainer.model, trainer.optimiser)

    # Set learning rate to zero so the model doesn't change while we're
    # collecting statistics about it.
    trainer.set_learning_rate(0.0)

    # Set up hooks to collect data.
    util.stats.gradient_distribution_hooks(trainer.model)

    # Train for one epoch
    trainer.train_epoch(epoch)
    
    # Output the results.
    with open(log, "w") as f:
        for line in util.stats.get_gradient_stats():
            f.write(line + "\n")


def generate_text(model, dataset, args):
    """
    Use the model to generate text. This text should look similar to the text
    that the model was trained on.

    :param model: trained language model to be used
    :param dataset: dataset the model was trained on
    :param args: arguments collected by `util.args.parse_args()`
    """
    model.eval()

    num_tokens = dataset.num_tokens()
    input_data = torch.randint(num_tokens, (1, 1), dtype=torch.long)

    if args.cuda:
        input_data = input_data.cuda(async=True)

    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output = model(input_data)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_data.fill_(word_idx)
            word = dataset.token_to_word(word_idx)

            print(word, end=" ")
        print()  # final new line
