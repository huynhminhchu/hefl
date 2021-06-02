from flib.FedsServer import FedServer
import sys

n = int(sys.argv[1])
server = FedServer(n_client=n)
server.wait()
