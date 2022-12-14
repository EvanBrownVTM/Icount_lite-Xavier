"""egohands_models.py
"""


SUPPORTED_MODELS = {
    'ssd_mobilenet_v1_egohands': {
        'config_path': 'data/ssd_mobilenet_v1_egohands.config',
        'checkpoint_path': 'data/ssd_mobilenet_v1_egohands/model.ckpt',
    },
    'ssd_mobilenet_v2_egohands': {
        'config_path': 'data/ssd_mobilenet_v2_egohands.config',
        'checkpoint_path': 'data/ssd_mobilenet_v2_egohands/model.ckpt-20000',
    },
    'ssdlite_mobilenet_v2_egohands': {
        'config_path': 'data/ssdlite_mobilenet_v2_egohands.config',
        'checkpoint_path': 'data/ssdlite_mobilenet_v2_egohands/model.ckpt-20000',
    },
    'ssd_resnet_50': {
        'config_path': 'data/ssd_resnet_50.config',
        'checkpoint_path': 'data/ssd_resnet_50/model.ckpt',
    },
    'ssd_mob_fpn': {
        'config_path': 'data/ssd_mob_fpn.config',
        'checkpoint_path': 'data/ssd_mob_fpn/model.ckpt',
    },
    'ssd_inception_v2_egohands': {
        'config_path': 'data/ssd_inception_v2_egohands.config',
        'checkpoint_path': 'data/ssd_inception_v2_egohands/model.ckpt-20000',
    },
    'pg_faster_rcnn': {
        'config_path': 'data/pg_faster_rcnn.config',
        'checkpoint_path': 'data/pg_faster_rcnn/model.ckpt',
    },
    'rfcn_resnet101_egohands': {
        'config_path': 'data/rfcn_resnet101_egohands.config',
        'checkpoint_path': 'data/rfcn_resnet101_egohands/model.ckpt-50000',
    },
    'faster_rcnn_resnet50_egohands': {
        'config_path': 'data/faster_rcnn_resnet50_egohands.config',
        'checkpoint_path': 'data/faster_rcnn_resnet50_egohands/model.ckpt-50000',
    },
    'faster_rcnn_resnet101_egohands': {
        'config_path': 'data/faster_rcnn_resnet101_egohands.config',
        'checkpoint_path': 'data/faster_rcnn_resnet101_egohands/model.ckpt-50000',
    },
    'faster_rcnn_inception_v2_egohands': {
        'config_path': 'data/faster_rcnn_inception_v2_egohands.config',
        'checkpoint_path': 'data/faster_rcnn_inception_v2_egohands/model.ckpt-50000',
    },
    'faster_rcnn_inception': {
        'config_path': 'data/faster_rcnn_inception.config',
        'checkpoint_path': 'data/faster_rcnn_inception/model.ckpt',
    },

}


def get_egohands_model(model_name):
    assert model_name in SUPPORTED_MODELS
    return (SUPPORTED_MODELS[model_name]['config_path'],
            SUPPORTED_MODELS[model_name]['checkpoint_path'])
