import * as React from 'react'

const useUSBDeviceConnected = (): boolean => {
  const [connected, setConnected] = React.useState(false)
  React.useEffect(() => {
    return () => { }
  }, [])
  return connected
}

const Read = () => {
  const connected = useUSBDeviceConnected()
  return (
    <>
      {connected ? 'connected' : 'not connected'}
    </>
  )
}

export default Read