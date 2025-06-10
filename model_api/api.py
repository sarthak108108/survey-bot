import os;
from dotenv import load_dotenv;

from flask import Flask, request, jsonify;
from flask_restful import Api, Resource, fields ,marshal_with , abort;
from werkzeug.utils import secure_filename;
import requests;

from cloudinary import uploader
import cloudinary

from app import Get_Predictions;

app = Flask(__name__)
api = Api(app)
app.config['upload_folder'] = os.path.join(app.root_path, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'jpeg'}
spring_boot_url = os.getenv('SPRING_BOOT_URL')

load_dotenv()
cloudinary_key=os.getenv('CLOUDINARY_KEY')
cloudinary_secret=os.getenv('CLOUDINARY_SECRET')
cloudinary_url=os.getenv('CLOUDINARY_URL')
cloudinary_cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME')

cloudinary.config(
    cloud_name=cloudinary_cloud_name,
    api_key=cloudinary_key,
    api_secret=cloudinary_secret,
    secure=True
)

BinFields = {
    'id':fields.Integer,
    'bin_level':fields.Float,
    'Image':fields.String,
    'label':fields.String
}

# reqparser = reqparse.RequestParser()
# reqparser.add_argument('bin_level', type=float, required=True, help='Bin level required')
# reqparser.add_argument('Image', type=str, required=False)

@app.route('/')
def home():
    return '<h1>Model Api Live</h1>'

class Model(Resource):
    @marshal_with(BinFields)
    def post(self, id):
        # args = reqparser.parse_args()

        if 'Image' not in request.files:
            abort(400, message="image not found")
        else:
            file = request.files['Image']
            if file.filename == '':
                abort(400, message='No file selected')
            filename = secure_filename(file.filename)
            # filename = file.filename
            filepath = os.path.join(app.config['upload_folder'], filename)
            try:
                file.save(filepath)
            except Exception as e:
                abort(400, message='File not saved')

        # bin_level = args[0]['bin_level']
        bin_level = float(request.form['bin_level'])
        if bin_level < 0 or bin_level > 100:
            abort(400, message='bin level should be between 0 and 100')

#validation ends here
#Model Call
        output_file_name, label_clean = Get_Predictions(filepath)
        if not output_file_name:
            abort(400, message='Model prediction failed')
        
#check for allowed labels
        trash_labels = {"umbrella","handbag", "tie", "sports ball", "kite", "bottle", "wine glass", "cup", "fork", "knife", "bowl", "spoon",
                          "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "toilet", "mouse",
                          "book", "clock", "vase", "scissors", "teddy bear", "toothbrush"}
#deleting file not in trash labels
        if label_clean not in trash_labels:
            os.remove(filepath)
            os.remove(output_file_name)
            return {
                'id': id,
                'bin_level': bin_level,
                'Image': "",
                'label': label_clean
            }
        
#output file uploaded to cloudinary
        try:
            output_url = uploader.upload(output_file_name)
        except Exception as e:
            abort(400, message='File not uploaded to cloudinary')
        os.remove(filepath)
        os.remove(output_file_name)

#spring-boot calls
    #post the results to spring endpoint 
    #check if upload was successful
    #return suitable status code and response
        try: 
            bin_save_request = requests.post(f"{spring_boot_url}/api/bins", json={
                'id': id,
                'bin_level': bin_level,
                'Image': output_url['secure_url'],
                'label': label_clean
            }, timeout=10)
            if bin_save_request.status_code != 200:
                bin_save_request.raise_for_status()
                abort(400, message="failed to update database!")
            else:
                return {
                    'id': id,
                    'bin_level': bin_level,
                    'Image': output_url['secure_url'],
                    'label': label_clean,
                    'status': f"success: {bin_save_request.status_code}"
                }
            
        except requests.exceptions.RequestException as e:
            abort(400, message="connection to db failed")
        #
        return {
            'id': id,
            'bin_level': bin_level,
            'Image': output_url['secure_url'],
            'label':label_clean
        }
    
#spring-boot-relay
@app.route('/relay/<path: endpoint>', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
def relay(endpoint):
    method = request.method
    url = f"{spring_boot_url}{endpoint}"
    data = request.get_json(silent=True)
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.request(method, url, json = data, headers = headers, timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
        
api.add_resource(Model, '/model/predict/<int:id>')
    

if __name__ == '__main__':
    app.run(debug=False)