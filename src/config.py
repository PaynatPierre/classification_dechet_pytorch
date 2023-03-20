from easydict import EasyDict as edict


__C                                             = edict()
# Consumers can get config by: from lstm_config import cfg

cfg                                             = __C

# DATASET options
__C.DATASET                                     = edict()

__C.DATASET.DATA_FOLDER_PATH                    = "./../data/"
__C.DATASET.TRAIN_PROPORTION                    = 0.8
__C.DATASET.VALIDATION_PROPORTION               = 0.1
__C.DATASET.TRAINING_BATCH_SIZE                 = 32
__C.DATASET.VALIDATION_BATCH_SIZE               = 32
__C.DATASET.TEST_BATCH_SIZE                     = 32
__C.DATASET.LABELS                              = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# TRAIN options
__C.TRAIN                                       = edict()

__C.TRAIN.LEARNING_RATE                         = 0.0005
__C.TRAIN.NBR_EPOCH                             = 300
__C.TRAIN.CHECKPOINT_SAVE_PATH                  = './../models/try_8/'
__C.TRAIN.VALIDATION_RATIO                      = 1
__C.TRAIN.GRADIANT_ACCUMULATION                 = 1
__C.TRAIN.IMAGE_SHAPE                           = (256,256)
__C.TRAIN.RESENET50_WEIGHTS                     = "IMAGENET1K_V2"

# EVALUATION options
__C.EVALUATION                                  = edict()

__C.EVALUATION.PRETRAINED_PATH                  = './../models/try_8/ckpt_96_metric_0.92497.ckpt'