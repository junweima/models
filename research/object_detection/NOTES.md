# Notes for Tensorflow Object Detection API

+ model_lib.py
    + create_estimator_and_inputs
        + calling:
            + model_fn_creator
            + detection_model_fn
    + create_model_fn
        + args
            * detection_model_fn (returns DetectionModel)
            * configs
            * hparams
        * returns: model_fn
        + model_fn
            * 
    + 


+ model_builder.py
    + build
        + args
            * model_config
            * is_training
            * add_summaries
            * add_background_class
        + returns: DetectionModel



+ python
    + functools.partial
        + # basically a way to pass default params
        + basetwo = paritial(int, base=2)
        + basetwo('10010') = 18




