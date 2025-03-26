from flask import Flask, render_template
from flask_cors import CORS
from flask_app.routes import routes  # Ensure routes.py file is correct and updated!

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register your blueprint with a prefix if needed (optional)
    app.register_blueprint(routes)
    
    @app.route('/')
    def index():
        return render_template("index.html")

    return app

# Initialize the app directly if running from here
app = create_app()
 