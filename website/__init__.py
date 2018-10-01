from flask import Flask
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, instance_relative_config=True)

from website import views

# db = SQLAlchemy(app)
