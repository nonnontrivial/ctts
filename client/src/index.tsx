import * as React from 'react'
import { createRoot } from 'react-dom/client'

const App = () => {
  console.log(42)
  return null
}

const root = createRoot(document.getElementById('root') as HTMLElement)
root.render(<App />)