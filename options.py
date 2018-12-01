class Options(object):
  pass

def get_config(is_train):
  opts = Options()
  if is_train:
    opts.batch_size = 300
    opts.isize = 32
    opts.lr = 1e-4
    opts.iteration = 15    
    opts.ckpt_dir = "ckpt"
    opts.z_size = 100
    opts.image_channel=1
    opts.dis_filter=64
    opts.gen_filter=64
    opts.n_extra_layers=0
    opts.test_batch_size = 300
  else:
    opts.batch_size = 5
    opts.im_size = [128, 128]

    opts.result_dir = "result"
    opts.ckpt_dir = "ckpt"
  return opts
