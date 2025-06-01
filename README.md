# Agentic AI Framework

This is a generalized framework for building agentic AI systems for internal question answering.


## Running service manually

To run the service manually in debug mode install the required python dependencies:

`uv install`

You can run the service in dev mode by default:

via fastapi app:
```
make dev
```
and access via [http://127.0.0.1:5055/docs](http://127.0.0.1:5055/docs)

via cli:

```
make run Q="What is the daily maximum value of PI-P0017 in April 2025?"
```

or for rag retrieval:
```
make run Q="How much was produced in the first two weeks of 2025?"
```

Examples:
```
- "What is the daily maximum value of PI-P0017 in April 2025?"
- "How much was produced in the first two weeks of 2025?"
- "Can you compare PI-P0017 and PI-P0016 for the first 10 days in 2025?"
- "What assets are next to asset BA100?"
- "Can you create a plot for the adjacent sensors of asset BA101 for 1st January 2025?"
- "What is the id of TI-T0022?"
- "What is the name of asset id c831fadb-d620-4007-bdda-4593038c87f9?"
- "Can you provide me the highest value for June 2025 for TI-T0022?"
- "How much was the total production in the first two weeks of 2025?"
- "How much was the total production in the distillation first two weeks of 2025?"
- "What is the current pressure in the distillation?"
- "What is the level in tank b?"
- "can you plot me the temperature of the distillation cooler A for the last two weeks?"
- "What is the current temperature in the water tank?"
- "how much distillate is flown in the storage in the last 4 hours?"
- Can you plot me data for 18b04353-839d-40a1-84c1-9b547d09dd80 in Febuary?

```

## Running service in Docker


```
make up
```

and to shut down the service:

```
make down
```


## Local querying


## API Documentation


## Testing

To run the tests:

`uv run python -m pytest --verbose --cov=./`


## Agent Design

![Agent Design](architecture/agent_design.png)


## System Design

![System Design](architecture/system_design.png)



This is a personal project inspired by my past work, but built independently from scratch. This project is oriented by this[book](https://www.cosmicpython.com/)
