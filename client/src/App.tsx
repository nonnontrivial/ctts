import * as React from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Root, { NotFound } from './routes/Root'
import Read from './routes/Read'
import View from './routes/View'

const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <NotFound />,
    children: [
      {
        path: "read",
        element: <Read />,
      },
      {
        path: "view",
        element: <View />,
      },
    ]
  },
])

const App = () => (
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)

export default App