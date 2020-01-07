from __future__ import division, print_function


def print_stage(**kwargs):
    stage = kwargs.pop('stage', 0)
    epoch = kwargs.pop('epoch', 0)
    it = kwargs.pop('it', 0)
    loss = kwargs.pop('loss', float('nan'))
    print('[{:02d} - {:02d} - {:05d}] {:.5f}'.format(stage, epoch, it, loss))


def print_step(**kwargs):
    epoch = kwargs.pop('epoch', 0)
    it = kwargs.pop('it', 0)
    loss = kwargs.pop('loss', float('nan'))
    val_loss = kwargs.pop('val_loss', float('nan'))
    print('[{:02d} - {:05d}] {:.5f} {:.5f}'.format(epoch, it, loss, val_loss))


def print_fn(iter_items=['epoch', 'it'], scalar_items=['loss', 'val_loss'], end='\n'):
    def inner(**kwargs):
        if 'it' in kwargs and kwargs['it'] == 0:
            print()
        print('[{iter:s}] {scalar:s}'.format(
            iter=' - '.join('{:05d}'.format(kwargs.pop(i, 0)) for i in iter_items),
            scalar=' '.join('{:.5f}'.format(kwargs.pop(s, float('nan'))) for s in scalar_items)),
            end=end)
    return inner

