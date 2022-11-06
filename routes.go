package main

import (
	"context"
	"log"

	pb "github.com/nonnontrivial/ctts/service"
	"google.golang.org/grpc"
)

type serviceServer struct {
	pb.UnimplementedServiceServer
}

func (ss *serviceServer) Read(ctx context.Context, in *pb.ReadRequest) (*pb.ReadReply, error) {
	log.Println(in.GetBrightness())
	return &pb.ReadReply{}, nil
}

func (ss *serviceServer) View(ctx context.Context, in *pb.ViewRequest) (*pb.ViewReply, error) {
	lat := in.GetLat()
	lng := in.GetLng()
	log.Println(lat, lng)
	return &pb.ViewReply{Brightness: "1"}, nil
}

// routes establishes the gRPC server and handles the REST-based endpoints.
func (s *server) routes() {
	g := grpc.NewServer()
	pb.RegisterServiceServer(g, &serviceServer{})
	if err := buildClient(); err != nil {
		log.Fatalln(err)
	}
	spa := spaHandler{clientFiles, "client/dist", "index.html"}
	s.router.Handle("/", spa)
}
