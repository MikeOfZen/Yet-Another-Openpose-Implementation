import tensorflow as tf
from config import *

CONV_BLOCK_NFILTERS=96
STAGE_FINAL_NFILTERS=256
BATCH_NORMALIZATION_ON=False

class ModelMaker():
    def __init__(self):
        pass


    def make_stage0(self, x):
        vgg_input_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=x)
        final_vgg_kayer = vgg_input_model.get_layer("block3_pool")
        input_model = tf.keras.Model(inputs=vgg_input_model.inputs, outputs=final_vgg_kayer.output)
        input_model.trainable = True

        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv1")(input_model.output)
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv2")(x)
        x = tf.keras.layers.Conv2D(256, 1, padding="same", activation='relu', name="stage0_final_conv3")(x)
        x = tf.keras.layers.Conv2D(128, 1, padding="same", activation='relu', name="stage0_final_conv4")(x)
        return x

    def make_conv_block(self,x, conv_block_filters, name):
        if BATCH_NORMALIZATION_ON: x = tf.keras.layers.BatchNormalization(name=name + "_bn3")(x)
        x1 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv1")(x)
        if BATCH_NORMALIZATION_ON: x1 = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x1)
        x2 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv2")(x1)
        if BATCH_NORMALIZATION_ON: x2 = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x2)
        x3 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv3")(x2)

        output = tf.keras.layers.concatenate([x1, x2, x3], name=name + "_output")
        return output

    def make_stageI(self,inputs, name, conv_block_filters, outputs):
        if len(inputs) > 1:
            x = tf.keras.layers.concatenate(inputs, name=name + "_input")
        else:
            x = inputs[0]
        x = self.make_conv_block(x, conv_block_filters, name + "_block1")
        x = self.make_conv_block(x, conv_block_filters, name + "_block2")
        x = self.make_conv_block(x, conv_block_filters, name + "_block3")
        x = self.make_conv_block(x, conv_block_filters, name + "_block4")
        x = self.make_conv_block(x, conv_block_filters, name + "_block5")

        x = tf.keras.layers.Conv2D(STAGE_FINAL_NFILTERS, 1, padding="same", activation='relu', name=name + "_final1conv")(x)
        if BATCH_NORMALIZATION_ON:
            x = tf.keras.layers.BatchNormalization(name=name + "_finalbn1")(x)
        x = tf.keras.layers.Conv2D(outputs, 1, padding="same", activation='relu', name=name + "_outputconv")(x)
        if BATCH_NORMALIZATION_ON:
            x = tf.keras.layers.BatchNormalization(name=name + "_finalbn2")(x)
        return x

    def make_model(self,mode):
        assert mode in ("train", "test")
        outputs = []

        input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        input_tensor = tf.keras.layers.Input(shape=input_shape)

        stage1_input = self.make_stage0(input_tensor)
        # PAF stages
        # stage 1
        stage1_output = self.make_stageI([stage1_input], "stage1paf", 96, PAF_OUTPUT_FILTERS)
        # stage 2
        stage2_output = self.make_stageI([stage1_output, stage1_input], "stage2paf", 128, PAF_OUTPUT_FILTERS)
        # stage 3
        stage3_output = self.make_stageI([stage2_output, stage1_input], "stage3paf", 128, PAF_OUTPUT_FILTERS)
        # stage 4
        stage4_output = self.make_stageI([stage3_output, stage1_input], "stage4paf", 128, PAF_OUTPUT_FILTERS)
        # keypoint heatmap stages
        # stage5
        stage5_output = self.make_stageI([stage4_output, stage1_input], "stage5heatmap", 96, HEATMAP_FILTERS)
        # stage6
        stage6_output = self.make_stageI([stage5_output, stage4_output, stage1_input], "stage6heatmap", 128, HEATMAP_FILTERS)

        if mode == "train":
            outputs = [stage1_output, stage2_output, stage3_output, stage4_output, stage5_output, stage6_output]
            output_types = ["PAF", "PAF", "PAF", "PAF", "HEATMAP", "HEATMAP"]

        if mode == "test":
            outputs = [stage4_output, stage6_output]
            output_types = ["PAF", "HEATMAP"]

        model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

        return model, output_types

