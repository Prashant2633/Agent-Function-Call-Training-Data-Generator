import json
from pathlib import Path

# Target directory
schemas_dir = Path("schemas")

# Helper to write schema to file
def write_schema(domain, name, desc, properties, required=None):
    if required is None:
        required = []
    
    schema_content = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "name": name,
        "domain": domain,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    
    domain_path = schemas_dir / domain
    domain_path.mkdir(parents=True, exist_ok=True)
    
    file_path = domain_path / f"{name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(schema_content, f, indent=2)
    print(f"Wrote schema: {domain}/{name}.json")

# 1. Finance (remaining)
write_schema(
    "finance", "get_portfolio_summary",
    "Retrieves a summary of an investment portfolio, including performance metrics and asset allocation.",
    {
        "portfolio_id": {
            "type": "string",
            "description": "Unique identifier of the portfolio.",
            "examples": ["port_12345", "p_abcde"],
            "$comment": "Opaque string ID linked to user account."
        },
        "as_of_date": {
            "type": "string",
            "description": "Optional date (ISO 8601 format) to retrieve the summary for. Defaults to current date.",
            "examples": ["2024-09-15", "2023-12-31"],
            "$comment": "ISO 8601 date YYYY-MM-DD."
        },
        "include_performance": {
            "type": "boolean",
            "description": "Whether to calculate and return historical performance metrics.",
            "default": True,
            "examples": [True, False],
            "$comment": "Set to false to reduce computation if only allocation is needed."
        },
        "include_allocation": {
            "type": "boolean",
            "description": "Whether to return asset allocation percentages by asset class or ticker.",
            "default": True,
            "examples": [True, False],
            "$comment": "Set to false to omit asset-by-asset weights."
        },
        "currency": {
            "type": "string",
            "description": "ISO 4217 three-letter currency code to display values.",
            "default": "USD",
            "examples": ["USD", "EUR", "JPY"],
            "$comment": "Standard currency conversion applies."
        }
    },
    ["portfolio_id"]
)

write_schema(
    "finance", "place_order",
    "Places a stock order (buy or sell) for a given ticker symbol.",
    {
        "ticker": {
            "type": "string",
            "description": "Stock ticker symbol to trade.",
            "examples": ["AAPL", "TSLA", "NVDA"],
            "$comment": "Case-insensitive ticker symbol."
        },
        "order_type": {
            "type": "string",
            "description": "Type of order to execute.",
            "enum": ["market", "limit", "stop", "stop_limit"],
            "examples": ["market", "limit"],
            "$comment": "Limit orders require price parameter; stop orders require stop_price."
        },
        "side": {
            "type": "string",
            "description": "Side of the transaction.",
            "enum": ["buy", "sell"],
            "examples": ["buy", "sell"],
            "$comment": "Indicates buy or sell order."
        },
        "quantity": {
            "type": "number",
            "description": "Number of shares to trade.",
            "minimum": 0.0001,
            "examples": [10, 100, 1.5],
            "$comment": "Allows fractional share trading."
        },
        "price": {
            "type": "number",
            "description": "Limit price. Required for limit and stop_limit orders.",
            "examples": [150.25, 200.00],
            "$comment": "Optional for market orders."
        },
        "stop_price": {
            "type": "number",
            "description": "Stop price. Required for stop and stop_limit orders.",
            "examples": [140.00, 190.50],
            "$comment": "Triggers the order if stock hits this price."
        },
        "time_in_force": {
            "type": "string",
            "description": "Time in force for the order.",
            "enum": ["day", "gtc", "ioc", "fok"],
            "default": "day",
            "examples": ["day", "gtc"],
            "$comment": "gtc = Good 'Til Cancelled."
        },
        "account_id": {
            "type": "string",
            "description": "Unique identifier of the trading account.",
            "examples": ["acc_998877", "brokerage_01"],
            "$comment": "Required to verify trading permissions."
        }
    },
    ["ticker", "order_type", "side", "quantity", "account_id"]
)

write_schema(
    "finance", "get_exchange_rate",
    "Retrieves the conversion rate between two currencies.",
    {
        "from_currency": {
            "type": "string",
            "description": "ISO 4217 three-letter source currency code.",
            "examples": ["USD", "EUR", "GBP"],
            "$comment": "Base currency."
        },
        "to_currency": {
            "type": "string",
            "description": "ISO 4217 three-letter target currency code.",
            "examples": ["JPY", "CAD", "AUD"],
            "$comment": "Target currency."
        },
        "amount": {
            "type": "number",
            "description": "Amount to convert. Defaults to 1.0.",
            "default": 1.0,
            "examples": [1.0, 100.00],
            "$comment": "Calculates exchange total for this amount."
        },
        "as_of_date": {
            "type": "string",
            "description": "Optional historical date (ISO 8601 format) to retrieve rates for.",
            "examples": ["2024-05-01", "2020-01-01"],
            "$comment": "ISO 8601 date YYYY-MM-DD."
        }
    },
    ["from_currency", "to_currency"]
)

write_schema(
    "finance", "calculate_roi",
    "Calculates return on investment (ROI) based on purchase price, current value, and holding period.",
    {
        "initial_investment": {
            "type": "number",
            "description": "The starting value or purchase cost of the investment.",
            "examples": [1000.00, 50000.00],
            "$comment": "Must be greater than 0."
        },
        "final_value": {
            "type": "number",
            "description": "The current or exit value of the investment.",
            "examples": [1250.00, 75000.00],
            "$comment": "Can be lower than initial_investment to represent a loss."
        },
        "holding_period_days": {
            "type": "integer",
            "description": "The number of days the investment was held.",
            "examples": [365, 180, 730],
            "$comment": "Used to calculate annualized ROI."
        },
        "include_annualized": {
            "type": "boolean",
            "description": "Whether to return the annualized ROI in addition to absolute ROI.",
            "default": True,
            "examples": [True, False],
            "$comment": "Annualized calculations require holding_period_days."
        },
        "dividends_received": {
            "type": "number",
            "description": "Optional dividends or interest payments received during holding period.",
            "default": 0.0,
            "examples": [50.00, 0.0],
            "$comment": "Added to final_value for net ROI."
        }
    },
    ["initial_investment", "final_value", "holding_period_days"]
)

# 2. Email
write_schema(
    "email", "send_email",
    "Sends an email with formatting, attachments, and scheduling options.",
    {
        "to": {
            "type": "array",
            "items": {"type": "string", "format": "email"},
            "description": "List of primary recipient email addresses.",
            "examples": [["boss@example.com"], ["user1@test.com", "user2@test.com"]],
            "$comment": "Array of strings validated as email format."
        },
        "cc": {
            "type": "array",
            "items": {"type": "string", "format": "email"},
            "description": "Optional list of carbon copy recipients.",
            "examples": [["cc@example.com"]],
            "$comment": "Optional cc addresses."
        },
        "bcc": {
            "type": "array",
            "items": {"type": "string", "format": "email"},
            "description": "Optional list of blind carbon copy recipients.",
            "examples": [["bcc@example.com"]],
            "$comment": "Optional bcc addresses."
        },
        "subject": {
            "type": "string",
            "description": "The subject line of the email.",
            "examples": ["Project status update", "Meeting agenda"],
            "$comment": "Plain string."
        },
        "body": {
            "type": "string",
            "description": "The main text body of the email.",
            "examples": ["Hi team, here is the status...", "Please review the attached..."],
            "$comment": "Multi-line string allowed."
        },
        "body_format": {
            "type": "string",
            "description": "The format of the email body.",
            "enum": ["plain", "html", "markdown"],
            "default": "plain",
            "examples": ["plain", "markdown"],
            "$comment": "HTML and markdown are parsed server-side."
        },
        "attachments": {
            "type": "array",
            "description": "Optional array of attachment objects containing name and download URL.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Filename of attachment.", "examples": ["invoice.pdf"]},
                    "url": {"type": "string", "description": "URL to download file.", "examples": ["https://storage.com/files/invoice.pdf"]}
                },
                "required": ["name", "url"]
            },
            "examples": [[{"name": "status.pdf", "url": "https://storage.com/status.pdf"}]]
        },
        "reply_to": {
            "type": "string",
            "format": "email",
            "description": "Optional email address to route replies to.",
            "examples": ["support@mycompany.com"]
        },
        "schedule_send": {
            "type": "string",
            "description": "Optional ISO 8601 datetime to schedule sending this email.",
            "examples": ["2024-09-15T08:00:00Z"],
            "$comment": "Requires timezone info."
        }
    },
    ["to", "subject", "body"]
)

write_schema(
    "email", "search_inbox",
    "Searches user's email inbox using query filters.",
    {
        "query": {
            "type": "string",
            "description": "Search keywords or filters (e.g. from:boss, has:attachment).",
            "examples": ["from:Alice billing", "meeting status"],
            "$comment": "Supports Gmail-style syntax."
        },
        "folder": {
            "type": "string",
            "description": "Email folder to search in.",
            "enum": ["inbox", "sent", "drafts", "spam", "trash", "all"],
            "default": "inbox",
            "examples": ["inbox", "all"]
        },
        "date_from": {
            "type": "string",
            "description": "Optional start date filter (YYYY-MM-DD).",
            "examples": ["2024-01-01"]
        },
        "date_to": {
            "type": "string",
            "description": "Optional end date filter (YYYY-MM-DD).",
            "examples": ["2024-06-30"]
        },
        "has_attachment": {
            "type": "boolean",
            "description": "Filter to messages that have attachments.",
            "examples": [True, False]
        },
        "is_read": {
            "type": "boolean",
            "description": "Filter to read/unread messages.",
            "examples": [True, False]
        },
        "limit": {
            "type": "integer",
            "description": "Max number of messages to return.",
            "minimum": 1,
            "maximum": 100,
            "default": 20,
            "examples": [10, 50]
        }
    },
    ["query"]
)

write_schema(
    "email", "mark_read",
    "Updates status flags of specific email messages.",
    {
        "message_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Array of unique message ID strings.",
            "examples": [["msg_1a2b3c", "msg_4d5e6f"]],
            "$comment": "Ids returned from search_inbox."
        },
        "mark_as": {
            "type": "string",
            "description": "Status flag to apply.",
            "enum": ["read", "unread", "starred", "important"],
            "examples": ["read", "starred"]
        },
        "folder": {
            "type": "string",
            "description": "Optional folder where messages are stored.",
            "examples": ["inbox", "archive"]
        }
    },
    ["message_ids", "mark_as"]
)

write_schema(
    "email", "get_thread",
    "Retrieves full message history for an email conversation thread.",
    {
        "thread_id": {
            "type": "string",
            "description": "Unique thread identifier.",
            "examples": ["th_998877"],
            "$comment": "Ids returned from search_inbox."
        },
        "include_attachments": {
            "type": "boolean",
            "description": "Whether to return details of attachments in thread.",
            "default": False,
            "examples": [True, False]
        },
        "max_messages": {
            "type": "integer",
            "description": "Optional maximum number of messages to return from the thread.",
            "examples": [5, 10]
        },
        "order": {
            "type": "string",
            "description": "Order of messages in thread.",
            "enum": ["asc", "desc"],
            "default": "asc",
            "examples": ["asc", "desc"],
            "$comment": "asc = oldest first, desc = newest first."
        }
    },
    ["thread_id"]
)

# 3. Files
write_schema(
    "files", "read_file",
    "Reads text file contents with custom limits and encodings.",
    {
        "path": {
            "type": "string",
            "description": "Absolute or relative file path to read.",
            "examples": ["data/reports.csv", "config.json"],
            "$comment": "Must exist on sandbox file system."
        },
        "encoding": {
            "type": "string",
            "description": "Character encoding system.",
            "enum": ["utf-8", "utf-16", "ascii", "binary"],
            "default": "utf-8",
            "examples": ["utf-8", "binary"]
        },
        "start_line": {
            "type": "integer",
            "description": "Line number (1-indexed) to start reading from.",
            "minimum": 1,
            "examples": [1, 100]
        },
        "end_line": {
            "type": "integer",
            "description": "Line number (1-indexed) to stop reading at.",
            "minimum": 1,
            "examples": [50, 200]
        },
        "format": {
            "type": "string",
            "description": "Output representation format.",
            "enum": ["raw", "base64"],
            "default": "raw",
            "examples": ["raw", "base64"],
            "$comment": "Use base64 for binary files."
        }
    },
    ["path"]
)

write_schema(
    "files", "write_file",
    "Writes text or binary content to a file.",
    {
        "path": {
            "type": "string",
            "description": "Target file path to write to.",
            "examples": ["output/result.txt", "notes.md"]
        },
        "content": {
            "type": "string",
            "description": "Text or base64 encoded string data to write.",
            "examples": ["Hello world!", "aGVsbG8gd29ybGQ="]
        },
        "encoding": {
            "type": "string",
            "description": "Target character encoding.",
            "default": "utf-8",
            "examples": ["utf-8", "ascii"]
        },
        "mode": {
            "type": "string",
            "description": "File write behavior.",
            "enum": ["overwrite", "append", "create_new"],
            "examples": ["overwrite", "append"]
        },
        "create_dirs": {
            "type": "boolean",
            "description": "Create parent directories recursively if they do not exist.",
            "default": True,
            "examples": [True, False]
        },
        "backup_existing": {
            "type": "boolean",
            "description": "Rename the existing file to .bak before overwriting.",
            "default": False,
            "examples": [True, False]
        }
    },
    ["path", "content", "mode"]
)

write_schema(
    "files", "list_directory",
    "Lists directory contents with glob pattern filtering.",
    {
        "path": {
            "type": "string",
            "description": "Directory path to list.",
            "examples": ["src/", "exports/temp"]
        },
        "recursive": {
            "type": "boolean",
            "description": "Whether to recursively list subdirectories.",
            "default": False,
            "examples": [True, False]
        },
        "pattern": {
            "type": "string",
            "description": "Optional glob filter pattern (e.g. *.json, **/*.py).",
            "examples": ["*.json", "**/*.csv"]
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Whether to list hidden files starting with .",
            "default": False,
            "examples": [True, False]
        },
        "sort_by": {
            "type": "string",
            "description": "Field to sort the file list by.",
            "enum": ["name", "size", "modified", "created"],
            "default": "name",
            "examples": ["name", "modified"]
        },
        "limit": {
            "type": "integer",
            "description": "Optional limit of file entries to return.",
            "examples": [50]
        }
    },
    ["path"]
)

write_schema(
    "files", "move_file",
    "Moves or renames a file or directory.",
    {
        "source_path": {
            "type": "string",
            "description": "Source path of file to move.",
            "examples": ["exports/draft.csv"]
        },
        "destination_path": {
            "type": "string",
            "description": "Target destination path.",
            "examples": ["exports/archive/draft.csv"]
        },
        "overwrite": {
            "type": "boolean",
            "description": "Whether to replace existing files at the destination.",
            "default": False,
            "examples": [True, False]
        },
        "create_dirs": {
            "type": "boolean",
            "description": "Whether to create missing directories in target path.",
            "default": True,
            "examples": [True, False]
        }
    },
    ["source_path", "destination_path"]
)

# 4. Notifications
write_schema(
    "notifications", "send_push",
    "Sends a mobile push notification to a user device.",
    {
        "user_id": {
            "type": "string",
            "description": "Unique user device mapping key.",
            "examples": ["user_12345", "u_abcde"]
        },
        "title": {
            "type": "string",
            "description": "Title line of the notification.",
            "examples": ["New message received", "Alert: low battery"]
        },
        "body": {
            "type": "string",
            "description": "Longer text body content.",
            "examples": ["John sent you a draft document.", "Battery at 10%."]
        },
        "data": {
            "type": "object",
            "description": "Optional key-value JSON payload sent with the notification.",
            "examples": [{"chat_id": "123", "action": "open"}]
        },
        "icon_url": {
            "type": "string",
            "description": "Optional URL pointing to thumbnail icon.",
            "examples": ["https://cdn.com/icon.png"]
        },
        "click_action": {
            "type": "string",
            "description": "Optional deep link routing screen name.",
            "examples": ["chat_screen", "home"]
        },
        "priority": {
            "type": "string",
            "description": "Message priority.",
            "enum": ["low", "normal", "high"],
            "default": "normal",
            "examples": ["high", "normal"]
        },
        "ttl_seconds": {
            "type": "integer",
            "description": "Time to live in seconds if device is offline.",
            "examples": [3600, 86400]
        }
    },
    ["user_id", "title", "body"]
)

write_schema(
    "notifications", "send_sms",
    "Sends a standard text SMS to a phone number.",
    {
        "to": {
            "type": "string",
            "description": "Recipient phone number in E.164 format.",
            "examples": ["+12125551234", "+447911123456"],
            "$comment": "E.164 international standard."
        },
        "message": {
            "type": "string",
            "description": "The message body text. Maximum 1600 characters.",
            "maxLength": 1600,
            "examples": ["Your code is 9876.", "Hello, your package has shipped."]
        },
        "from_number": {
            "type": "string",
            "description": "Optional sender shortcode or phone number.",
            "examples": ["+18885550199"]
        },
        "schedule_at": {
            "type": "string",
            "description": "Optional ISO 8601 datetime to schedule SMS.",
            "examples": ["2024-09-15T08:00:00Z"]
        },
        "is_unicode": {
            "type": "boolean",
            "description": "Whether to force Unicode encoding (enables emoji).",
            "default": False,
            "examples": [True, False]
        }
    },
    ["to", "message"]
)

write_schema(
    "notifications", "send_slack_message",
    "Sends a message or block structure to a Slack channel.",
    {
        "channel": {
            "type": "string",
            "description": "Target channel name or channel ID (e.g. #general, C12345).",
            "examples": ["#alerts", "C998877"],
            "$comment": "IDs preferred for stable routing."
        },
        "text": {
            "type": "string",
            "description": "Plain text fallback message.",
            "examples": ["Deployment successful!"]
        },
        "blocks": {
            "type": "array",
            "description": "Optional Block Kit visual structure array.",
            "items": {"type": "object"},
            "examples": [[{"type": "section", "text": {"type": "mrkdwn", "text": "Deploying..."}}]]
        },
        "thread_ts": {
            "type": "string",
            "description": "Optional parent message timestamp to reply in thread.",
            "examples": ["1712345678.000100"]
        },
        "username": {
            "type": "string",
            "description": "Optional override display name for webhook.",
            "examples": ["Release Bot"]
        },
        "icon_emoji": {
            "type": "string",
            "description": "Optional Slack emoji string (e.g. :rocket:).",
            "examples": [":rocket:", ":warning:"]
        },
        "attachments": {
            "type": "array",
            "description": "Optional legacy attachments list.",
            "items": {"type": "object"},
            "examples": [[{"color": "#36a64f", "text": "Green alert"}]]
        }
    },
    ["channel", "text"]
)

# 5. Maps
write_schema(
    "maps", "get_directions",
    "Retrieves route directions between origin and destination.",
    {
        "origin": {
            "type": "string",
            "description": "Starting location address or coordinate pair.",
            "examples": ["New York Penn Station", "40.7128,-74.0060"]
        },
        "destination": {
            "type": "string",
            "description": "Ending location address or coordinate pair.",
            "examples": ["Empire State Building", "40.7484,-73.9857"]
        },
        "mode": {
            "type": "string",
            "description": "Travel transit type.",
            "enum": ["driving", "walking", "cycling", "transit"],
            "default": "driving",
            "examples": ["driving", "transit"]
        },
        "waypoints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional intermediate addresses to stop at.",
            "examples": [["Times Square", "Bryant Park"]]
        },
        "avoid": {
            "type": "array",
            "items": {"type": "string", "enum": ["tolls", "highways", "ferries"]},
            "description": "Optional road elements to route around.",
            "examples": [["tolls", "highways"]]
        },
        "departure_time": {
            "type": "string",
            "description": "Optional ISO 8601 departure time to estimate traffic.",
            "examples": ["2024-09-15T17:30:00"]
        },
        "units": {
            "type": "string",
            "description": "Measurement standard.",
            "enum": ["metric", "imperial"],
            "default": "metric",
            "examples": ["metric", "imperial"]
        }
    },
    ["origin", "destination"]
)

write_schema(
    "maps", "find_nearby",
    "Searches for places of a specific type in a radius around a location.",
    {
        "location": {
            "type": "string",
            "description": "Center location address or coordinate pair.",
            "examples": ["Times Square, NYC", "34.0522,-118.2437"]
        },
        "place_type": {
            "type": "string",
            "description": "Category of place to find.",
            "enum": [
                "restaurant", "cafe", "hospital", "pharmacy", "hotel", 
                "gas_station", "atm", "supermarket", "school", "parking"
            ],
            "examples": ["restaurant", "atm"]
        },
        "radius_meters": {
            "type": "integer",
            "description": "Search radius in meters.",
            "minimum": 100,
            "maximum": 50000,
            "default": 1000,
            "examples": [1000, 5000]
        },
        "min_rating": {
            "type": "number",
            "description": "Optional minimum user rating (0-5 stars).",
            "examples": [4.0, 4.5]
        },
        "open_now": {
            "type": "boolean",
            "description": "Filter to places currently open.",
            "default": False,
            "examples": [True, False]
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of places to return.",
            "minimum": 1,
            "maximum": 20,
            "default": 10,
            "examples": [5, 20]
        }
    },
    ["location", "place_type"]
)

write_schema(
    "maps", "calculate_distance",
    "Calculates distance and travel times between multiple origins and destinations.",
    {
        "origins": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of origin addresses or coordinate pairs.",
            "examples": [["NYC Penn Station", "Grand Central Terminal"]]
        },
        "destinations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of destination addresses or coordinate pairs.",
            "examples": [["Times Square, NYC", "Central Park, NYC"]]
        },
        "mode": {
            "type": "string",
            "description": "Travel transit type.",
            "enum": ["driving", "walking", "cycling", "transit"],
            "default": "driving",
            "examples": ["driving", "transit"]
        },
        "units": {
            "type": "string",
            "description": "Measurement standard.",
            "enum": ["metric", "imperial"],
            "default": "metric",
            "examples": ["metric", "imperial"]
        },
        "departure_time": {
            "type": "string",
            "description": "Optional traffic estimation departure time.",
            "examples": ["2024-09-15T08:00:00"]
        }
    },
    ["origins", "destinations"]
)

# 6. Tasks
write_schema(
    "tasks", "create_task",
    "Creates a new task in a project management system.",
    {
        "title": {
            "type": "string",
            "description": "Brief title summarizing the task.",
            "examples": ["Fix login bug", "Draft Q3 newsletter"]
        },
        "description": {
            "type": "string",
            "description": "Detailed explanation of task requirements.",
            "examples": ["User fails to login when using special characters..."]
        },
        "assignee_id": {
            "type": "string",
            "description": "Optional user ID assigned to complete the task.",
            "examples": ["usr_887766"]
        },
        "due_date": {
            "type": "string",
            "description": "Optional ISO 8601 date YYYY-MM-DD for task completion.",
            "examples": ["2024-09-20"]
        },
        "priority": {
            "type": "string",
            "description": "Task urgency tier.",
            "enum": ["low", "medium", "high", "critical"],
            "default": "medium",
            "examples": ["high", "critical"]
        },
        "project_id": {
            "type": "string",
            "description": "Optional parent project category identifier.",
            "examples": ["proj_backend"]
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional labels to categorize the task.",
            "examples": [["bug", "auth"]]
        },
        "parent_task_id": {
            "type": "string",
            "description": "Optional parent task ID for subtasks.",
            "examples": ["task_112233"]
        }
    },
    ["title"]
)

write_schema(
    "tasks", "assign_task",
    "Assigns or delegates a task to a user.",
    {
        "task_id": {
            "type": "string",
            "description": "Unique identifier of target task.",
            "examples": ["task_998877"]
        },
        "assignee_id": {
            "type": "string",
            "description": "Unique identifier of assignee user.",
            "examples": ["usr_887766"]
        },
        "notify": {
            "type": "boolean",
            "description": "Whether to send an notification email/alert to assignee.",
            "default": True,
            "examples": [True, False]
        },
        "message": {
            "type": "string",
            "description": "Optional custom delegation note sent to assignee.",
            "examples": ["Hey, assigning this bug to you as you wrote the login code."]
        },
        "due_date": {
            "type": "string",
            "description": "Optional new or modified due date YYYY-MM-DD.",
            "examples": ["2024-09-30"]
        }
    },
    ["task_id", "assignee_id"]
)

write_schema(
    "tasks", "update_status",
    "Updates progress status and logs comments on a task.",
    {
        "task_id": {
            "type": "string",
            "description": "Unique identifier of target task.",
            "examples": ["task_998877"]
        },
        "status": {
            "type": "string",
            "description": "New workflow status.",
            "enum": ["todo", "in_progress", "blocked", "review", "done", "cancelled"],
            "examples": ["in_progress", "done"]
        },
        "comment": {
            "type": "string",
            "description": "Optional status update rationale note.",
            "examples": ["Blocked waiting for API credentials..."]
        },
        "updated_fields": {
            "type": "object",
            "description": "Optional dict tracking other modified fields (e.g. completion percentage).",
            "examples": [{"percent_complete": 50}]
        }
    },
    ["task_id", "status"]
)

write_schema(
    "tasks", "get_overdue",
    "Queries task list for incomplete items past their due date.",
    {
        "assignee_id": {
            "type": "string",
            "description": "Optional assignee user ID filter.",
            "examples": ["usr_887766"]
        },
        "project_id": {
            "type": "string",
            "description": "Optional project category filter.",
            "examples": ["proj_frontend"]
        },
        "days_overdue": {
            "type": "integer",
            "description": "Only return tasks overdue by at least this many days.",
            "default": 0,
            "examples": [0, 7]
        },
        "include_completed": {
            "type": "boolean",
            "description": "Whether to include completed tasks.",
            "default": False,
            "examples": [True, False]
        },
        "limit": {
            "type": "integer",
            "description": "Optional limit of tasks in return list.",
            "default": 20,
            "examples": [20, 50]
        },
        "sort_by": {
            "type": "string",
            "description": "Sorting criteria.",
            "enum": ["due_date", "priority", "created"],
            "default": "due_date",
            "examples": ["due_date", "priority"]
        }
    },
    []
)

# 7. Database
write_schema(
    "database", "query_records",
    "Performs filter queries on database table records.",
    {
        "table": {
            "type": "string",
            "description": "Target database table name.",
            "examples": ["users", "transactions", "orders"]
        },
        "filters": {
            "type": "object",
            "description": "Filter criteria key-value object (exact matches).",
            "examples": [{"status": "active", "tier": "premium"}]
        },
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional columns array to return. Defaults to all columns.",
            "examples": [["id", "name", "email"]]
        },
        "limit": {
            "type": "integer",
            "description": "Max rows to return.",
            "minimum": 1,
            "maximum": 1000,
            "default": 100,
            "examples": [50, 500]
        },
        "offset": {
            "type": "integer",
            "description": "Query pagination offset.",
            "default": 0,
            "examples": [0, 100]
        },
        "order_by": {
            "type": "string",
            "description": "Column name to order results by.",
            "examples": ["created_at", "id"]
        },
        "order_direction": {
            "type": "string",
            "description": "Sorting ordering direction.",
            "enum": ["asc", "desc"],
            "default": "asc",
            "examples": ["asc", "desc"]
        }
    },
    ["table"]
)

write_schema(
    "database", "insert_record",
    "Inserts a new record into a table.",
    {
        "table": {
            "type": "string",
            "description": "Target database table name.",
            "examples": ["users", "contacts"]
        },
        "data": {
            "type": "object",
            "description": "Column-value key mappings for the new row.",
            "examples": [{"name": "Jane Doe", "email": "jane@example.com"}]
        },
        "return_id": {
            "type": "boolean",
            "description": "Whether to return the auto-generated primary key ID.",
            "default": True,
            "examples": [True, False]
        },
        "on_conflict": {
            "type": "string",
            "description": "Action if conflict on unique constraint occurs.",
            "enum": ["error", "ignore", "update"],
            "default": "error",
            "examples": ["error", "ignore"]
        }
    },
    ["table", "data"]
)

write_schema(
    "database", "update_record",
    "Updates records matching filter conditions with new field values.",
    {
        "table": {
            "type": "string",
            "description": "Target database table name.",
            "examples": ["users", "orders"]
        },
        "filters": {
            "type": "object",
            "description": "Filter conditions identifying rows to update.",
            "examples": [{"id": 42}, {"status": "pending"}]
        },
        "updates": {
            "type": "object",
            "description": "New column-value pairs to apply.",
            "examples": [{"status": "completed", "updated_at": "2024-09-15"}]
        },
        "limit": {
            "type": "integer",
            "description": "Optional max rows limit to apply update to.",
            "examples": [1]
        },
        "return_updated": {
            "type": "boolean",
            "description": "Whether to return the updated record list.",
            "default": False,
            "examples": [True, False]
        }
    },
    ["table", "filters", "updates"]
)

write_schema(
    "database", "delete_record",
    "Deletes rows matching filter conditions from a table.",
    {
        "table": {
            "type": "string",
            "description": "Target database table name.",
            "examples": ["logs", "sessions"]
        },
        "filters": {
            "type": "object",
            "description": "Filter criteria. For safety, empty filter objects are rejected.",
            "examples": [{"session_id": "sess_112233"}, {"user_id": "usr_99887"}]
        },
        "soft_delete": {
            "type": "boolean",
            "description": "If true, flags record as deleted_at=now instead of physical deletion.",
            "default": False,
            "examples": [True, False],
            "$comment": "Requires deleted_at column on table."
        },
        "return_deleted": {
            "type": "boolean",
            "description": "Whether to return the list of deleted record identifiers.",
            "default": False,
            "examples": [True, False]
        }
    },
    ["table", "filters"]
)

print("Finished writing all missing schemas!")
