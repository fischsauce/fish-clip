import mimetypes
from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse

import sys
import base64
# sys.path.append("clipit")
# sys.path.append("../../clipit")
# sys.path.append("CLIP")
# sys.path.append("../../CLIP")
# sys.path.append("diffvg")
# sys.path.append("../../diffvg")
# sys.path.append("taming-transformers")
# sys.path.append("../../taming-transformers")

import clipit

serve = Flask(__name__)
api = Api(serve)

class Inference(Resource):
# @serve.route('/inference', methods='GET')
    def get(self):
        args = request.args

        if (not args['prompts'] or len(args['prompts']) == 0):
            return {
                'message': 'Prompt string must not be empty'
            }, 400

        prompts = args['prompts']
        print(f"Received request with prompts: {prompts}")

        

        use_pixeldraw = True #@param {type:"boolean"}

       
        clipit.reset_settings()
        clipit.add_settings(prompts=prompts, size=[256, 256])
        clipit.add_settings(quality="normal", iterations=10, pixel_scale=0.5)
        clipit.add_settings(use_pixeldraw=use_pixeldraw)

        settings = clipit.apply_settings()

        clipit.do_init(settings)
        clipit.do_run(settings)



        # with open("output.png", "rb") as image_file:
        #     encoded_string = base64.b64encode(image_file.read())

        # return {"image": encoded_string, "data": jsonify(return_list)}


        return send_file('../output.png', mimetype='image/png')


        # return {'data': data}, 200  # return data and 200 OK code

api.add_resource(Inference, '/inference')


if __name__ == '__main__':
    serve.run()