import * as React from 'react'
import { Outlet } from "react-router-dom"

const Root = () => {
  const [errorMessage, setErrorMesage] = React.useState<string | null>(null)
  React.useEffect(() => {
    const geoEnabled = 'geolocation' in window.navigator
    if (!geoEnabled) {
      window.alert('geolocation not enabled!')
    }
    try {
      window.navigator.geolocation.getCurrentPosition((pos) => {
        console.log(pos)
        setErrorMesage(null)
      })
    } catch {
      setErrorMesage('something went wrong')
    }
  }, [])

  return (
    <>
      {errorMessage}
      <Outlet />
    </>
  )
}

export default Root