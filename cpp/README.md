# cpp

producer for rabbitmq queue which handles sky brightness prediction across h3 cells.

it finds cells to request predictions for, requests those predictions, and then sends
the response to the prediction queue (repeatedly)
