import * as React from 'react'

const App = () => {
  const [errorMessage, setErrorMesage] = React.useState<string | null>(null)
  const [geoEnabled, setGeoEnabled] = React.useState(false)
  React.useEffect(() => {
    setGeoEnabled('geolocation' in window.navigator)
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
      geolocation enabled: {geoEnabled ? 'true' : 'false'}
    </>
  )
}

export default App