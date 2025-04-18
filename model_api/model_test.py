import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class WasteClassifier:
    def __init__(self, model_path=None):
        self.waste_categories = {

            'plastic_bottle': 'waste',
            'hard_disc': 'waste',
            'water_bottle': 'waste',
            'plastic_bag': 'waste',
            'cup': 'waste',
            'can': 'waste',
            'beer_bottle': 'waste',
            'wine_bottle': 'waste',
            'pop_bottle': 'waste',
            'paper': 'waste',
            'cardboard': 'waste',
            'packet': 'waste',
            'envelope': 'waste',
            'garbage': 'waste',
            'trash': 'waste',
            'waste_bin': 'waste',
            'trash_can': 'waste',
            'dumpster': 'waste',
            'glass': 'waste',
            'container': 'waste',
            'straw': 'waste',
            'carton': 'waste',
            'food_waste': 'waste',
            'banana_peel': 'waste',
            'apple_core': 'waste',
            

            'person': 'non-waste',
            'car': 'non-waste',
            'tree': 'non-waste',
            'building': 'non-waste',
            'chair': 'non-waste',
            'table': 'non-waste',
            'computer': 'non-waste',
            'phone': 'non-waste',
            'dog': 'non-waste',
            'cat': 'non-waste',
            'bicycle': 'non-waste',
            'book': 'non-waste',
            'television': 'non-waste',
            'sofa': 'non-waste',
            'house': 'non-waste'
        }
        

        if model_path:

            self.model = tf.keras.models.load_model(model_path)

            self.preprocess = preprocess_input

            self.use_imagenet_labels = False
        else:
            self.model = MobileNetV2(weights='imagenet')
            self.preprocess = preprocess_input
            self.use_imagenet_labels = True
    
    def _load_and_prepare_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return self.preprocess(img_array)
    
    def classify_image(self, img_path):
        processed_img = self._load_and_prepare_image(img_path)
        
        predictions = self.model.predict(processed_img)
        
        if self.use_imagenet_labels:
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            top_prediction = decoded_predictions[0]
            category_id, category_name, confidence = top_prediction
            
            is_waste = False
            matching_categories = []
            
            for _, label_name, label_conf in decoded_predictions:
                clean_label = label_name.replace('_', ' ').lower()
                
                if label_name in self.waste_categories:
                    matching_categories.append((label_name, label_conf, self.waste_categories[label_name]))
                    if self.waste_categories[label_name] == 'waste':
                        is_waste = True
                
                else:
                    for category, category_type in self.waste_categories.items():
                        if category.replace('_', ' ') in clean_label:
                            matching_categories.append((label_name, label_conf, category_type))
                            if category_type == 'waste':
                                is_waste = True
                            break
            
            if not matching_categories:
                waste_classification = 'non-waste'
            else:
                matching_categories.sort(key=lambda x: x[1], reverse=True)
                waste_classification = matching_categories[0][2]
                
            return {
                'image_path': img_path,
                'top_category': category_name,
                'confidence': float(confidence),
                'top_5_predictions': [(label, float(conf)) for _, label, conf in decoded_predictions],
                'waste_classification': waste_classification,
                'is_waste': waste_classification == 'waste',
                'matching_categories': matching_categories
            }
        else:
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class_idx])
            
            category_name = f"class_{predicted_class_idx}"
            
            waste_classification = self.waste_categories.get(category_name, 'non-waste')
            
            return {
                'image_path': img_path,
                'predicted_class': predicted_class_idx,
                'category': category_name,
                'confidence': confidence,
                'waste_classification': waste_classification,
                'is_waste': waste_classification == 'waste'
            }


def predict(image_path, model_path=None):
    classifier = WasteClassifier(model_path=model_path)
    
    if not os.path.exists(image_path):
        return {
            'error': f"Image file not found: {image_path}"
        }
    
    try:
        result = classifier.classify_image(image_path)
        return result
    
    except Exception as e:
        return {
            'error': f"Error processing image: {str(e)}",
            'image_path': image_path
        }


result = predict("test1.jpg")
print(result)
