#!/bin/bash

# Script to set up Databricks secrets for OAuth service principal authentication
# Make sure you have the Databricks CLI installed and configured

echo "Setting up Databricks secrets for OAuth service principal authentication..."

# Create the secrets scope if it doesn't exist
echo "Creating secrets scope 'contextual_advertising'..."
databricks secrets create-scope contextual_advertising

# Set the service principal ID
echo "Setting service principal ID..."
read -p "Enter your Service Principal ID: " SP_ID
databricks secrets put-secret contextual_advertising service_principal_id --string-value "$SP_ID"

# Set the service principal secret
echo "Setting service principal secret..."
read -s -p "Enter your Service Principal Secret: " SP_SECRET
echo
databricks secrets put-secret contextual_advertising service_principal_secret --string-value "$SP_SECRET"

# Set the endpoint name
echo "Setting endpoint name..."
read -p "Enter your Model Serving Endpoint name: " ENDPOINT_NAME
databricks secrets put-secret contextual_advertising endpoint_name --string-value "$ENDPOINT_NAME"

echo "✅ All OAuth secrets have been configured!"
echo "You can now deploy your MCP server app with OAuth authentication." 