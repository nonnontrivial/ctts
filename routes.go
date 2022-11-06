package main

import (
	"context"
	"log"

	pb "github.com/nonnontrivial/ctts/service"
	"google.golang.org/grpc"
)

type gServer struct {
	pb.UnimplementedServiceServer
}

func (gs *gServer) Read(ctx context.Context, in *pb.ReadRequest) (*pb.ReadReply, error) {
	log.Println(in.GetBrightness())
	return &pb.ReadReply{}, nil
}

func (gs *gServer) View(ctx context.Context, in *pb.ViewRequest) (*pb.ViewReply, error) {
	lat := in.GetLat()
	lng := in.GetLng()
	log.Println(lat, lng)
	return &pb.ViewReply{Brightness: "1"}, nil
}

func (s *server) routes() {
	g := grpc.NewServer()
	pb.RegisterServiceServer(g, &gServer{})

	if err := buildClient(); err != nil {
		log.Fatalln(err)
	}
	spa := spaHandler{clientFiles, "client/dist", "index.html"}
	s.router.PathPrefix("/").Handler(spa)
}
