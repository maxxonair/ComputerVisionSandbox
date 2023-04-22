"""
This function starts and runs the camera web server


"""
from flask import Flask, render_template, Response
import cv2 as cv 


from util.PyLog import PyLog

app = Flask(__name__)

# Create logging instance
log = PyLog()

# Flag, if true -> Start and run camera stream
enableCameraStream = True

def generateFrameByFrame(): 
    if enableCameraStream:
        # Create OpenCV VideoCapture instance for webcam at port 0
        camera = cv.VideoCapture(0)  
        while True:
            # Capture frame-by-frame
            success, frame = camera.read()
            if not success:
                log.pLogErr('Failed to conntect to camera.')
                break
            else:
                ret, buffer = cv.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generateFrameByFrame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)