package main

import (
	"context"
	"log"
	"time"

	pb "github.com/nonnontrivial/ctts/proto/service"
	"google.golang.org/grpc"
)

type serviceServer struct {
	pb.UnimplementedServiceServer
}

func (ss *serviceServer) Read(ctx context.Context, in *pb.ReadRequest) (*pb.ReadReply, error) {
	return &pb.ReadReply{}, nil
}

type profile struct {
	lat, lng string
	timeOf   time.Time
}

func (ss *serviceServer) View(ctx context.Context, in *pb.ViewRequest) (*pb.ViewReply, error) {
	p := &profile{
		lat:    in.GetLat(),
		lng:    in.GetLng(),
		timeOf: time.Now(),
	}
	log.Println(p)
	return &pb.ViewReply{Brightness: "1"}, nil
}

func (s *server) routes() {
	g := grpc.NewServer()
	pb.RegisterServiceServer(g, &serviceServer{})
	if err := buildClient(); err != nil {
		log.Fatalln(err)
	}
	spa := spaHandler{clientFiles, "client/dist", "index.html"}
	s.router.Handle("/", spa)
}
