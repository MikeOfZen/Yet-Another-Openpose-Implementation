import tensorflow as tf
from config import IMAGE_HEIGHT,IMAGE_WIDTH,PAF_NUM_FILTERS,HEATMAP_NUM_FILTERS,BATCH_NORMALIZATION_ON



class ModelMaker():
    """Creates a model for the OpenPose project, structre is 10 layers of VGG16 followed by a few convultions, and 6 stages 
    of (PAF,PAF,PAF,PAF,kpts,kpts)"""
    
    def __init__(self):
        self.image_height=IMAGE_HEIGHT
        self.image_width=IMAGE_WIDTH
        self.paf_output_num_filters=PAF_NUM_FILTERS
        self.heatmap_num_filters=HEATMAP_NUM_FILTERS
        
        self.conv_block_nfilters = 96
        self.stage_final_nfilters = 256
        self.batch_normalization_on = BATCH_NORMALIZATION_ON

    def _make_vgg_input_model(self,input_tensor,input_shape):             
     
        vgg_input_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        tf.keras.backend.clear_session() #to clean to backaend from the imported model
        
        x=input_tensor

        name_last_layer = "block3_pool"
        for layer in vgg_input_model.layers[1:]:          
            config=layer.get_config()
            weights=layer.get_weights()
            #print(config)
            copy_layer=type(layer).from_config(config)  #the only way to make .from_config work
            
            x=copy_layer(x) #required for the proper sizing of the layer, set_weights will not work without it
            copy_layer.set_weights(weights)
            
            if layer.name == name_last_layer:
                break
                
        del vgg_input_model
        
        return x
        #input_model=tf.keras.Model(inputs=input_layer,outputs=x)

    def _make_stage0(self, x):
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv1")(x)
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
        x = tf.keras.layers.Conv2D(outputs, 1, padding="same", activation='relu', name=name + "_output")(x)
        if self.batch_normalization_on: x = tf.keras.layers.BatchNormalization(name=name + "_outputbn")(x)
        return x

    def create_models(self):
        tf.keras.backend.clear_session() #to clear previoius graphs
        
        input_shape = (self.image_height, self.image_width, 3)
        input_tensor = tf.keras.layers.Input(shape=input_shape) #first layer of the model
        #stage 00 (i know)
        stage00_output=self._make_vgg_input_model(input_tensor,input_shape)       
        #stage 0 2conv)
        stage0_output = self._make_stage0(stage00_output)
        # PAF stages
        # stage 1
        stage1_output = self._make_stageI([stage0_output], "stage1paf", 96, self.paf_output_num_filters)
        # stage 2
        stage2_output = self._make_stageI([stage1_output, stage0_output], "stage2paf", 128, self.paf_output_num_filters)
        # stage 3
        stage3_output = self._make_stageI([stage2_output, stage0_output], "stage3paf", 128, self.paf_output_num_filters)
        # stage 4
        stage4_output = self._make_stageI([stage3_output, stage0_output], "stage4paf", 128, self.paf_output_num_filters)
        # keypoint heatmap stages
        # stage5
        stage5_output = self._make_stageI([stage4_output, stage0_output], "stage5heatmap", 96, self.heatmap_num_filters)
        # stage6
        stage6_output = self._make_stageI([stage5_output, stage4_output, stage0_output], "stage6heatmap", 128, self.heatmap_num_filters)

        training_outputs = [stage1_output, stage2_output, stage3_output, stage4_output, stage5_output, stage6_output]
        train_model = tf.keras.Model(inputs=input_tensor, outputs=training_outputs)

        test_outputs = [stage4_output, stage6_output]
        test_model = tf.keras.Model(inputs=input_tensor, outputs=test_outputs)

        return train_model,test_model

