param(
    [string]$CsvPath = ".\evaluation\fixtures\employees.csv"
)

$ErrorActionPreference = "Stop"
Write-Host "QueryForge demo checklist"
Write-Host "1. Start the backend and Streamlit using README.md commands."
Write-Host "2. Open Database Explorer and upload: $CsvPath"
Write-Host "3. Open AI SQL Studio, generate SQL, inspect/edit, then Execute SQL."
Write-Host "4. View the result chart and download CSV."
Write-Host "5. Open Model Comparison for the same question."
Write-Host "6. Run: python -m evaluation.runner --limit 3"
Write-Host "7. Open Model Intelligence and select the saved run."
