import sys
from pathlib import Path
import zmq
import zmq.asyncio
from client_model import ClientModelDetectron2
from config import Config
from networking import EncoderPickle
sys.path.append(str(Path.home()) + '/detectron2')


def run(port):
    app = ClientModelDetectron2(Config())
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind("tcp://*:%s" % port)
    except zmq.error.ZMQError:
        print('cannot establish the service at port {}\n'.format(port))
    print('start to running service at port {}\n'.format(port))

    encoder = EncoderPickle()

    while True:
        arr = socket.recv()

        img = encoder.decode(arr)

        predictions = app.run(img)

        predictions['instances'] = predictions['instances'].to('cpu')

        socket.send_pyobj(predictions, copy=False)


if __name__ == "__main__":
    config = Config()
    run(config.server[1][1])
