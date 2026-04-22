from flask import Flask, jsonify
from db import get_user_by_username
import json

app = Flask(__name__)

@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    user = get_user_by_username(username)
    if user:
        return jsonify(dict(user))
    return jsonify({"error": "User not found"}), 404

@app.route('/config', methods=['GET'])
def load_config():
    config_data = {"status": "default_config"}
    try:
        with open('config.json', 'r') as f:
            config_data = json.load(f)
    except:
        pass
    
    return jsonify(config_data)

if __name__ == '__main__':
    app.run(debug=True)