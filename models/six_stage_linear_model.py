import tensorflow as tf
from config import IMAGE_HEIGHT,IMAGE_WIDTH,PAF_OUTPUT_NUM_FILTERS,HEATMAP_NUM_FILTERS,BATCH_NORMALIZATION_ON



class ModelMaker():
    def __init__(self):
        self.image_height=IMAGE_HEIGHT
        self.image_width=IMAGE_WIDTH
        self.paf_output_num_filters=PAF_OUTPUT_NUM_FILTERS
        self.heatmap_num_filters=HEATMAP_NUM_FILTERS
        
        self.conv_block_nfilters = 96
        self.stage_final_nfilters = 256
        self.batch_normalization_on = BATCH_NORMALIZATION_ON

    def _make_stage0(self, x):
        vgg_input_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=x)
        final_vgg_kayer = vgg_input_model.get_layer("block3_pool")
        input_model = tf.keras.Model(inputs=vgg_input_model.inputs, outputs=final_vgg_kayer.output)
        input_model.trainable = True

        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv1")(input_model.output)
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv2")(x)
        x = tf.keras.layers.Conv2D(256, 1, padding="same", activation='relu', name="stage0_final_conv3")(x)
        x = tf.keras.layers.Conv2D(128, 1, padding="same", activation='relu', name="stage0_final_conv4")(x)
        return x

    def _make_conv_block(self, x, conv_block_filters, name):
        if self.batch_normalization_on: x = tf.keras.layers.BatchNormalization(name=name + "_bn3")(x)
        x1 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv1")(x)
        if self.batch_normalization_on: x1 = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x1)
        x2 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv2")(x1)
        if self.batch_normalization_on: x2 = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x2)
        x3 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv3")(x2)

        output = tf.keras.layers.concatenate([x1, x2, x3], name=name + "_output")
        return output

    def _make_stageI(self, inputs, name, conv_block_filters, outputs):
        if len(inputs) > 1:
            x = tf.keras.layers.concatenate(inputs, name=name + "_input")
        else:
            x = inputs[0]
        x = self._make_conv_block(x, conv_block_filters, name + "_block1")
        x = self._make_conv_block(x, conv_block_filters, name + "_block2")
        x = self._make_conv_block(x, conv_block_filters, name + "_block3")
        x = self._make_conv_block(x, conv_block_filters, name + "_block4")
        x = self._make_conv_block(x, conv_block_filters, name + "_block5")

        x = tf.keras.layers.Conv2D(self.stage_final_nfilters, 1, padding="same", activation='relu', name=name + "_final1conv")(x)
        if self.batch_normalization_on: x = tf.keras.layers.BatchNormalization(name=name + "_finalbn1")(x)
        x = tf.keras.layers.Conv2D(outputs, 1, padding="same", activation='relu', name=name + "_outputconv")(x)
        if self.batch_normalization_on: x = tf.keras.layers.BatchNormalization(name=name + "_finalbn2")(x)
        return x

    def create_models(self):
        input_shape = (self.image_height, self.image_width, 3)
        input_tensor = tf.keras.layers.Input(shape=input_shape)

        #stage 0 (VGG-first 10 layers+ 2conv)
        stage1_input = self._make_stage0(input_tensor)
        # PAF stages
        # stage 1
        stage1_output = self._make_stageI([stage1_input], "stage1paf", 96, self.paf_output_num_filters)
        # stage 2
        stage2_output = self._make_stageI([stage1_output, stage1_input], "stage2paf", 128, self.paf_output_num_filters)
        # stage 3
        stage3_output = self._make_stageI([stage2_output, stage1_input], "stage3paf", 128, self.paf_output_num_filters)
        # stage 4
        stage4_output = self._make_stageI([stage3_output, stage1_input], "stage4paf", 128, self.paf_output_num_filters)
        # keypoint heatmap stages
        # stage5
        stage5_output = self._make_stageI([stage4_output, stage1_input], "stage5heatmap", 96, self.heatmap_num_filters)
        # stage6
        stage6_output = self._make_stageI([stage5_output, stage4_output, stage1_input], "stage6heatmap", 128, self.heatmap_num_filters)

        training_outputs = [stage1_output, stage2_output, stage3_output, stage4_output, stage5_output, stage6_output]
        self.train_model = tf.keras.Model(inputs=input_tensor, outputs=training_outputs)
        self.training_output_types = ["PAF", "PAF", "PAF", "PAF", "HEATMAP", "HEATMAP"]

        test_outputs = [stage4_output, stage6_output]
        self.test_model = tf.keras.Model(inputs=input_tensor, outputs=test_outputs)
        self.test_output_types = ["PAF", "HEATMAP"]

    def get_train_model(self):
        return self.train_model

    def get_test_model(self):
        return self.test_model

