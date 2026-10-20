# pip install grpcio
# pip install grpcio-tools

all: 
.PHONY: all

SIFTER := frb_sifter_pb2.py frb_sifter_pb2.pyi frb_sifter_pb2_grpc.py
all: $(SIFTER)

SIFTER_CHIME := frb_sifter_chime_pb2.py frb_sifter_chime_pb2.pyi frb_sifter_chime_pb2_grpc.py
all: $(SIFTER_CHIME)

SEARCH := frb_search_pb2.py frb_search_pb2.pyi frb_search_pb2_grpc.py
all: $(SEARCH)

# frb_sifter: set the package name in the generated python files.
$(SIFTER): frb_sifter.proto
	(cd .. && python -m grpc_tools.protoc -Ichord_frb_grpc=chord_frb_grpc --python_out=. --pyi_out=. --grpc_python_out=. chord_frb_grpc/$<)

# frb_search: set the package name in the generated python files.
$(SEARCH): frb_search.proto
	(cd .. && python -m grpc_tools.protoc -Ichord_frb_grpc=chord_frb_grpc --python_out=. --pyi_out=. --grpc_python_out=. chord_frb_grpc/$<)

%_pb2.py %_pb2.pyi %_pb2_grpc.py: %.proto
	python3 -m grpc_tools.protoc -I=. --python_out=. --pyi_out=. --grpc_python_out=. $<

PROTOC = protoc
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= $(shell which $(GRPC_CPP_PLUGIN))

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc %.grpc.pb.h: %.proto
	$(PROTOC) --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

.PRECIOUS: %.pb.cc
%.pb.cc %.pb.h: %.proto
	$(PROTOC) --cpp_out=. $<

clean:
	rm $(HELLO) $(SIFTER) $(SEARCH)
.PHONE: clean
