<#
.SYNOPSIS
This scripts runs the CCEW pipeline in a docker container with the run command. A running process can be stopped by calling the stop command.
You can peek into the current logs with the logs subcommand.  

.DESCRIPTION
USAGE
    .\run_pipeline.ps1 [<command>]

PARAMETERS
    -logs, -l   Directly peek into the logs when running the pipeline within the container.
    -test, -t   Run the pipeline in test mode. Only applies to the "predict" command.

COMMANDS
    predict     Run the prediction pipeline within the docker container.
    evaluate    Run the evaluation pipeline within the docker container.
    logs        Peek into the containers logs.
    stop        Stop the container from running the pipeline code.
    help        See this help message.
#>

param(
  [Parameter(Position=0, Mandatory=$False)]
  [ValidateSet("", "help", "predict", "evaluate", "logs", "stop")]
  [string]$Command,
  [switch]$logs = $false,
  [switch]$test = $false
)
function Command-Help { Get-Help $PSCommandPath }

$CMD_ARGS = ""
if ($test) {
    $CMD_ARGS = "-t"
}

function Start-Predict {
    Write-Host "[::] Running pipeline in Docker environment..."
    $image_id = docker images -q ccew-tree
    if ( !$image_id ) {
        # If the image doesn't exist yet, rebuild it.
        Write-Host "[::] CCEW image doesn't exist. Building image first..."
        docker build -t ccew-tree . 
    }
    Write-Host "[::] Running competition pipeline code in detached docker mode..."
    $container_id=$(docker ps -q --filter "ancestor=ccew-tree" |  Select-Object -First 1)
    if ($container_id) {
        Write-Host "[::] Container already running..."
    } else {
        $container_id=$(docker run -d -v ${PWD}:/usr/src/app ccew-tree "python3.11 -u competition_pipeline.py $CMD_ARGS")
    }
    if ($logs) {
        docker logs $container_id -f
        exit 0
    }
}

function Start-Evaluate {
    Write-Host "[::] Running the evaluation pipeline in Docker environment..."
    $image_id = docker images -q ccew-tree
    if ( !$image_id ) {
        # If the image doesn't exist yet, rebuild it.
        Write-Host "[::] CCEW image doesn't exist. Building image first..."
        docker build -t ccew-tree . 
    }
    Write-Host "[::] Running evaluatuion pipeline code in detached docker mode..."
    $container_id=$(docker ps -q --filter "ancestor=ccew-tree" |  Select-Object -First 1)
    if ($container_id) {
        Write-Host "[::] Container already running..."
    } else {
        $container_id=$(docker run -d -v ${PWD}:/usr/src/app ccew-tree "python3.11 -u evaluation_pipeline.py $CMD_ARGS")
    }
    if ($logs) {
        docker logs $container_id -f
        exit 0
    }
}

function Debug-Container {
    $container_id=$(docker ps -q --filter "ancestor=ccew-tree" |  Select-Object -First 1)
    if ($container_id) {
        docker logs $container_id -f
        exit 0
    } else {
        Write-Host "[::] No container is running. Start script in docker mode first!"
        exit 1
    }
}

function Stop-Container {
    $container_id=$(docker ps -q --filter "ancestor=ccew-tree" |  Select-Object -First 1)
    if ($container_id) {
        Write-Host "[::] Stopping docker container..."
        docker stop $container_id
    } else {
        Write-Host "[::] No CCEW pupeline docker container running. Do nothing..."
    }
}

if (!$Command) {
    Command-Help
    exit
}

switch ($Command) {
    "predict" { Start-Predict }
    "evaluate" { Start-Evaluate }
    "logs" { Debug-Container }
    "stop" { Stop-Container }
    "help"  { Command-Help }
}
