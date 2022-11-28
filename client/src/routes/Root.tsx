import * as React from 'react'
import { Outlet, Link, useRouteError } from "react-router-dom"

const NotFound = () => {
  const error = useRouteError()
  console.error(error)
  return (
    <>
      <p>that's an error (!)</p>
      <Link to="/">go back</Link>
    </>
  )
}

const Root = () => {
  return (
    <Outlet />
  )
}

export default Root
export { NotFound }