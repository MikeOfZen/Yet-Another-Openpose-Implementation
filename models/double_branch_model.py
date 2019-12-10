import tensorflow as tf
from config import IMAGE_HEIGHT,IMAGE_WIDTH,PAF_NUM_FILTERS,HEATMAP_NUM_FILTERS,BATCH_NORMALIZATION_ON,INCLUDE_MASK,LABEL_HEIGHT,LABEL_WIDTH


INPUT_SHAPE=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
MASK_SHAPE=(LABEL_HEIGHT, LABEL_WIDTH, 1)

class ModelMaker():
    """Creates a model for the OpenPose project, structre is 10 layers of VGG16 followed by a few convultions, and 6 stages 
    of (PAF,PAF,PAF,PAF,kpts,kpts) also potentially includes a mask stacked with the outputs"""
    
    def __init__(self):
        self.conv_block_nfilters = 96
        self.stage_final_nfilters = 256

        self._get_vgg_layer_config_weights()

    def _get_vgg_layer_config_weights(self):
        vgg_input_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
        name_last_layer = "block3_pool"
        
        self.vgg_layers=[]
        
        for layer in vgg_input_model.layers[1:]:
            layer_info={
                "config":layer.get_config()            
                ,"weights":layer.get_weights()  
                ,"type":type(layer)
            }
            self.vgg_layers.append(layer_info)
            if layer.name == name_last_layer:
                break               
        del vgg_input_model
        
    def _make_vgg_input_model(self,x):           
        for layer_info in self.vgg_layers:               
            copy_layer=layer_info["type"].from_config(layer_info["config"])  #the only way to make .from_config work            
            x=copy_layer(x) #required for the proper sizing of the layer, set_weights will not work without it
            copy_layer.set_weights(layer_info["weights"])                      
        return x

    def _make_stage0(self, x):
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv1")(x)
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv2")(x)
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv3")(x)
        x = tf.keras.layers.Conv2D(512, 1, padding="same", activation='relu', name="stage0_final_conv4")(x)
        return x

    def _make_conv_block(self, x, conv_block_filters, name):
        if BATCH_NORMALIZATION_ON: x = tf.keras.layers.BatchNormalization(name=name + "_bn3")(x)
        x1 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv1")(x)
        if BATCH_NORMALIZATION_ON: x1 = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x1)
        x2 = tf.keras.layers.Conv2D(conv_block_filters, 3, padding="same", activation='relu', name=name + "_conv2")(x1)
        if BATCH_NORMALIZATION_ON: x2 = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x2)
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
        if BATCH_NORMALIZATION_ON: x = tf.keras.layers.BatchNormalization(name=name + "_finalbn1")(x)
        x = tf.keras.layers.Conv2D(outputs, 1, padding="same", activation='relu', name=name + "_output")(x)
        if BATCH_NORMALIZATION_ON: x = tf.keras.layers.BatchNormalization(name=name + "_outputbn")(x)
        return x

    @staticmethod
    def _psd_zero_mask_to_outputs(outputs,mask_input):
        new_outputs=[]
        for i,output in enumerate(outputs):
            name=output.name.split("/")[0]+"_mask"
            new_outputs.append(
                tf.keras.layers.concatenate([output,mask_input],axis=-1,name=name)  #concat the mask to the output, at idx 0
            )
        return new_outputs

    def create_models(self):        
        input_tensor = tf.keras.layers.Input(shape=INPUT_SHAPE) #first layer of the model

        #mask_string="_pre_mask" if INCLUDE_MASK else ""

        #stage 00 (i know)
        stage00_output=self._make_vgg_input_model(input_tensor)       
        #stage 0 2conv)
        stage0_output = self._make_stage0(stage00_output)
        # PAF stages
        # stage 1
        s1paf_output = self._make_stageI([stage0_output], "s1paf", 128, PAF_NUM_FILTERS)
        s1kpt_output = self._make_stageI([stage0_output], "s1kpt", 128, HEATMAP_NUM_FILTERS)
        # stage 2
        s2paf_output = self._make_stageI([stage0_output,s1paf_output,s1kpt_output], "s2paf", 128, PAF_NUM_FILTERS)
        s2kpt_output = self._make_stageI([stage0_output,s1paf_output,s1kpt_output], "s2kpt", 128, HEATMAP_NUM_FILTERS)
        # stage 3
        s3paf_output = self._make_stageI([stage0_output,s2paf_output,s2kpt_output], "s3paf", 128, PAF_NUM_FILTERS)
        s3kpt_output = self._make_stageI([stage0_output,s2paf_output,s2kpt_output], "s3kpt", 128, HEATMAP_NUM_FILTERS)        
        # stage 4
        s4paf_output = self._make_stageI([stage0_output,s3paf_output,s3kpt_output], "s4paf", 128, PAF_NUM_FILTERS)
        s4kpt_output = self._make_stageI([stage0_output,s3paf_output,s3kpt_output], "s4kpt", 128, HEATMAP_NUM_FILTERS)             
        
        
        training_inputs=input_tensor
        training_outputs = [s1paf_output,s1kpt_output,
                           s2paf_output,s2kpt_output,
                           s3paf_output,s3kpt_output,
                           s4paf_output,s4kpt_output]

        if INCLUDE_MASK:  #this is used to pass the mask directly to the loss function through the model
            mask_input= tf.keras.layers.Input(shape=MASK_SHAPE)
            training_outputs=self._psd_zero_mask_to_outputs(training_outputs,mask_input)
            training_inputs=(input_tensor,mask_input)

        train_model = tf.keras.Model(inputs=training_inputs, outputs=training_outputs)

        test_outputs = [s4paf_output,s4kpt_output]
        test_model = tf.keras.Model(inputs=input_tensor, outputs=test_outputs)

        return train_model,test_model
   
@tf.function
def place_training_labels(elem):
    """Distributes labels into the correct configuration for the model, ie 4 PAF stage, 2 kpt stages
    must match the model"""
    paf_tr = elem['paf']
    kpt_tr = elem['kpts']
    image = elem['image']

    if INCLUDE_MASK:
        inputs = (image,elem['mask'])
    else:
        inputs = image

    return inputs, (paf_tr, kpt_tr,
                    paf_tr, kpt_tr,
                    paf_tr, kpt_tr,
                    paf_tr, kpt_tr)  # this should match the model outputs, and is different for each model
