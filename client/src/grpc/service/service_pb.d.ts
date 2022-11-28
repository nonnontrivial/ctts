import * as jspb from 'google-protobuf'



export class ReadRequest extends jspb.Message {
  getBrightness(): string;
  setBrightness(value: string): ReadRequest;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ReadRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ReadRequest): ReadRequest.AsObject;
  static serializeBinaryToWriter(message: ReadRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ReadRequest;
  static deserializeBinaryFromReader(message: ReadRequest, reader: jspb.BinaryReader): ReadRequest;
}

export namespace ReadRequest {
  export type AsObject = {
    brightness: string,
  }
}

export class ReadReply extends jspb.Message {
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ReadReply.AsObject;
  static toObject(includeInstance: boolean, msg: ReadReply): ReadReply.AsObject;
  static serializeBinaryToWriter(message: ReadReply, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ReadReply;
  static deserializeBinaryFromReader(message: ReadReply, reader: jspb.BinaryReader): ReadReply;
}

export namespace ReadReply {
  export type AsObject = {
  }
}

export class ViewRequest extends jspb.Message {
  getLat(): string;
  setLat(value: string): ViewRequest;

  getLng(): string;
  setLng(value: string): ViewRequest;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ViewRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ViewRequest): ViewRequest.AsObject;
  static serializeBinaryToWriter(message: ViewRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ViewRequest;
  static deserializeBinaryFromReader(message: ViewRequest, reader: jspb.BinaryReader): ViewRequest;
}

export namespace ViewRequest {
  export type AsObject = {
    lat: string,
    lng: string,
  }
}

export class ViewReply extends jspb.Message {
  getBrightness(): string;
  setBrightness(value: string): ViewReply;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ViewReply.AsObject;
  static toObject(includeInstance: boolean, msg: ViewReply): ViewReply.AsObject;
  static serializeBinaryToWriter(message: ViewReply, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ViewReply;
  static deserializeBinaryFromReader(message: ViewReply, reader: jspb.BinaryReader): ViewReply;
}

export namespace ViewReply {
  export type AsObject = {
    brightness: string,
  }
}

