from flask import Flask, render_template,request, send_file

import functions as f

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('main.html')  # Replace 'UI.html' with the name of your HTML file




@app.route('/upload', methods=['POST'])
def upload():
    #############getting data from ui
    data = request.get_json() 
    numComponents=int(data['numComponents'])
    varianceThreshold=data['varianceThreshold']
    selectedOption = data['selectedOption']
    ###############################################################################
    ################      LOGIC     ################################
    
    
    
    
    f.eigenFaces(numComponents,varianceThreshold,selectedOption)
   
   
    ####################################################################################
    

    # Return the processed image file
    return send_file('generated_image.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)