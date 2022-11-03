import * as React from 'react'
import { createBrowserRouter, RouterProvider, useRouteError } from 'react-router-dom'
import Root from './routes/Root'
import Read from './routes/Read'

const NotFound = () => {
  const error = useRouteError()
  console.error(error)

  return (
    <>
      <p>that's an error (!)</p>
    </>
  )
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <NotFound />,
    children: [
      {
        path: "read",
        element: <Read />,
      }
    ]
  },
])

const App = () => (
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)

export default App