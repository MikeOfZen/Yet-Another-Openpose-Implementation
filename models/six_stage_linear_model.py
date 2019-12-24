import tensorflow as tf


class ModelMaker:
    """Creates a model for the OpenPose project, structure is 10 layers of VGG16 followed by a few convolutions, and 6 stages
    of (PAF,PAF,PAF,PAF,kpts,kpts) also potentially includes a mask stacked with the outputs"""

    def __init__(self, config):
        self.IMAGE_HEIGHT = config.IMAGE_HEIGHT
        self.IMAGE_WIDTH = config.IMAGE_WIDTH
        self.PAF_NUM_FILTERS = config.PAF_NUM_FILTERS
        self.HEATMAP_NUM_FILTERS = config.HEATMAP_NUM_FILTERS
        self.BATCH_NORMALIZATION_ON = config.BATCH_NORMALIZATION_ON
        self.DROPOUT_RATE = config.DROPOUT_RATE

        self.INCLUDE_MASK = config.INCLUDE_MASK
        self.LABEL_HEIGHT = config.LABEL_HEIGHT
        self.LABEL_WIDTH = config.LABEL_WIDTH
        self.INPUT_SHAPE = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)
        self.MASK_SHAPE = (config.LABEL_HEIGHT, config.LABEL_WIDTH, 1)

        self.stage_final_nfilters = 512
        self.base_activation = tf.keras.layers.PReLU
        self.base_activation_kwargs = {'shared_axes': [1, 2]}

        self._get_vgg_layer_config_weights()

    def _get_vgg_layer_config_weights(self):
        vgg_input_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=self.INPUT_SHAPE)
        name_last_layer = "block3_pool"

        self.vgg_layers = []

        for layer in vgg_input_model.layers[1:]:
            layer_info = {
                    "config"   : layer.get_config()
                    , "weights": layer.get_weights()
                    , "type"   : type(layer)
                    }
            self.vgg_layers.append(layer_info)
            if layer.name == name_last_layer:
                break
        del vgg_input_model

    def _make_vgg_input_model(self, x):
        for layer_info in self.vgg_layers:
            copy_layer = layer_info["type"].from_config(layer_info["config"])  # the only way to make .from_config work
            x = copy_layer(x)  # required for the proper sizing of the layer, set_weights will not work without it
            copy_layer.set_weights(layer_info["weights"])
        return x

    def _make_stage0(self, x):
        x = tf.keras.layers.Conv2D(512, 1, padding="same", name="stage0_final_conv1")(x)
        x = self.base_activation(**self.base_activation_kwargs, name="stage0_final_conv1_act")(x)
        x = tf.keras.layers.Conv2D(512, 1, padding="same", name="stage0_final_conv2")(x)
        x = self.base_activation(**self.base_activation_kwargs, name="stage0_final_conv2_act")(x)
        x = tf.keras.layers.Conv2D(256, 1, padding="same", name="stage0_final_conv3")(x)
        x = self.base_activation(**self.base_activation_kwargs, name="stage0_final_conv3_act")(x)
        x = tf.keras.layers.Conv2D(256, 1, padding="same", name="stage0_final_conv4")(x)
        x = self.base_activation(**self.base_activation_kwargs, name="stage0_final_conv4_act")(x)
        return x

    def _make_conv_block(self, x0, conv_block_filters, name):
        if self.BATCH_NORMALIZATION_ON: x0 = tf.keras.layers.BatchNormalization(name=name + "_bn3")(x0)
        x1 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", name=name + "_conv1")(x0)
        x1 = self.base_activation(**self.base_activation_kwargs, name=name + "_conv1_act")(x1)

        if self.BATCH_NORMALIZATION_ON: x1 = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x1)
        x2 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", name=name + "_conv2")(x1)
        x2 = self.base_activation(**self.base_activation_kwargs, name=name + "_conv2_act")(x2)

        if self.BATCH_NORMALIZATION_ON: x2 = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x2)
        x3 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", name=name + "_conv3")(x2)
        x3 = self.base_activation(**self.base_activation_kwargs, name=name + "_conv3_act")(x3)

        output = tf.keras.layers.concatenate([x1, x2, x3], name=name + "_output")
        return output

    def _make_stage_i(self, inputs, name, conv_block_filters, outputs, last_activation):
        if len(inputs) > 1:
            x = tf.keras.layers.concatenate(inputs, name=name + "_input")
        else:
            x = inputs[0]
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block1")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block2")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block3")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block4")
        if self.DROPOUT_RATE > 0: x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)
        x = self._make_conv_block(x, conv_block_filters, name + "_block5")

        x = tf.keras.layers.Conv2D(self.stage_final_nfilters, 1, padding="same", name=name + "_final1conv")(x)
        x = self.base_activation(**self.base_activation_kwargs, name=name + "_final1conv_act")(x)
        x = tf.keras.layers.Conv2D(outputs, 1, padding="same", activation=last_activation, name=name + "_final2conv")(x)

        return x

    @staticmethod
    def rename_outputs(pre_outputs):
        new_outputs = []
        for pre_output in pre_outputs:
            new_outputs.append(
                    tf.keras.layers.Lambda(lambda x: x, name=pre_output.name.split("_")[0] + "_output")(pre_output)
                    )
        return new_outputs

    @staticmethod
    def _psd_zero_mask_to_outputs(outputs, mask_input):
        new_outputs = []
        for i, output in enumerate(outputs):
            name = output.name.split("/")[0] + "_mask"
            new_outputs.append(
                    tf.keras.layers.concatenate([output, mask_input], axis=-1, name=name)  # concat the mask to the output, at idx 0
                    )
        return new_outputs

    def create_models(self):
        input_tensor = tf.keras.layers.Input(shape=self.INPUT_SHAPE)  # first layer of the model

        # mask_string="_pre_mask" if INCLUDE_MASK else ""

        # stage 00 (i know)
        stage00_output = self._make_vgg_input_model(input_tensor)
        # stage 0 2conv)
        stage0_output = self._make_stage0(stage00_output)
        # PAF stages
        # stage 1
        stage1_output = self._make_stage_i([stage0_output], "s1pafs", 96, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # stage 2
        stage2_output = self._make_stage_i([stage1_output, stage0_output], "s2pafs", 128, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # stage 3
        stage3_output = self._make_stage_i([stage2_output, stage0_output], "s3pafs", 128, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # stage 4
        stage4_output = self._make_stage_i([stage3_output, stage0_output], "s4pafs", 128, self.PAF_NUM_FILTERS, tf.keras.activations.linear)
        # keypoint heatmap stages
        # stage5
        stage5_output = self._make_stage_i([stage4_output, stage0_output], "s5kpts", 96, self.HEATMAP_NUM_FILTERS, tf.keras.activations.tanh)
        # stage6
        stage6_output = self._make_stage_i([stage5_output, stage4_output, stage0_output], "s6kpts", 128, self.HEATMAP_NUM_FILTERS, tf.keras.activations.tanh)

        training_inputs = input_tensor
        training_outputs = [stage1_output, stage2_output, stage3_output, stage4_output, stage5_output, stage6_output]

        if self.INCLUDE_MASK:  # this is used to pass the mask directly to the loss function through the model
            mask_input = tf.keras.layers.Input(shape=self.MASK_SHAPE)
            training_outputs = self._psd_zero_mask_to_outputs(training_outputs, mask_input)
            training_inputs = (input_tensor, mask_input)

        training_outputs = self.rename_outputs(training_outputs)

        train_model = tf.keras.Model(inputs=training_inputs, outputs=training_outputs)

        test_outputs = [stage4_output, stage6_output]
        test_model = tf.keras.Model(inputs=input_tensor, outputs=test_outputs)

        return train_model, test_model


class ModelDatasetComponent:
    def __init__(self, config):
        self.INCLUDE_MASK = config.INCLUDE_MASK

    @tf.function
    def place_training_labels(self, elem):
        """Distributes labels into the correct configuration for the model, ie 4 PAF stage, 2 kpt stages
        must match the model"""
        paf_tr = elem['pafs']
        kpt_tr = elem['kpts']
        image = elem['image']

        if self.INCLUDE_MASK:
            inputs = (image, elem['mask'])
        else:
            inputs = image
        return inputs, (paf_tr, paf_tr, paf_tr, paf_tr, kpt_tr, kpt_tr)  # this should match the model outputs, and is different for each model
