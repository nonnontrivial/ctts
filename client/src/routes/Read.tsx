import * as React from 'react'
import { Dialog } from '@headlessui/react'

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
      <Dialog open={!connected} onClose={() => { }} className="relative z-50">
        <Dialog.Panel>
          <Dialog.Title>USB device not connected</Dialog.Title>
        </Dialog.Panel>
      </Dialog>
    </>
  )
}

export default Read