import zmq.green as zmq


def connect_socket(ip, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{}:{}".format(ip, port))
    return socket


def connect_sockets(servers):
    return [
        connect_socket(ip, port) for ip, port in servers
    ]


def bind_socket(port):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    return socket


def close_sockets(sockets):
    for socket in sockets:
        socket.close()
        socket.context.destroy()
