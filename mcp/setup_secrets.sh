#!/bin/bash

# Script to set up Databricks secrets for authentication
# Make sure you have the Databricks CLI installed and configured

echo "Setting up Databricks secrets for authentication..."

# Create the secrets scope if it doesn't exist
echo "Creating secrets scope 'contextual_advertising'..."
databricks secrets create-scope contextual_advertising

# Set the Databricks personal access token
echo "Setting Databricks personal access token..."
read -s -p "Enter your Databricks Personal Access Token: " DATABRICKS_TOKEN
echo
databricks secrets put-secret contextual_advertising databricks_token --string-value "$DATABRICKS_TOKEN"

# Set the endpoint name
echo "Setting endpoint name..."
read -p "Enter your Model Serving Endpoint name: " ENDPOINT_NAME
databricks secrets put-secret contextual_advertising endpoint_name --string-value "$ENDPOINT_NAME"

echo "✅ All secrets have been configured!"
echo "You can now deploy your MCP server app." 