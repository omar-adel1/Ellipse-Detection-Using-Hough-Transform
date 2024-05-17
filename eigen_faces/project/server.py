from flask import Flask, render_template,request, jsonify, send_file
import io
import cv2
import os
import functions as f

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('main.html')  # Replace 'UI.html' with the name of your HTML file




@app.route('/upload', methods=['POST'])
def upload():
    #############getting data from ui
    data = request.get_json() 
    numComponents=data['numComponents']
    varianceThreshold=data['varianceThreshold']
    selectedOption = data['selectedOption']
    ###############################################################################
    ################      LOGIC     ################################
   
   
   
   
   
   
   
    ####################################################################################
    output_path = os.path.join(os.path.dirname(__file__), 'image.png')
    cv2.imwrite(output_path, output_image)

    # Return the processed image file
    return send_file('image.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)