import * as React from 'react'
import { useLoaderData, LoaderFunction } from "react-router-dom"

export const loader: LoaderFunction = ({ request, params }) => {
  console.log(request,params)
  return {}
}

const View = () => {
  const d = useLoaderData()
  console.log(d)
  return (
    <>
    </>
  )
}

export default View