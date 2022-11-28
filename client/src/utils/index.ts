import { endpoint } from "../constants/grpc"
import { ViewReply, ViewRequest } from "../grpc/service/service_grpc_web_pb"
import { ServicePromiseClient } from "../grpc/service/service_grpc_web_pb"

export const getBrightnessFromRequest = async (request: Request): Promise<ViewReply> => {
  const serviceClient = new ServicePromiseClient(endpoint, null, null)
  const [lat, lng] = ["", ""]
  const req = new ViewRequest().setLat(lat).setLng(lng)
  const res: ViewReply = await serviceClient.view(req, {})
  return res
}
