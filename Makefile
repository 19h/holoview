# Makefile for holographic-city-viewer
# Builds the project in release mode and code signs with get-task-pid entitlement

# Variables
BINARY_NAME = holographic-city-viewer
TARGET_DIR = target/release
BINARY_PATH = $(TARGET_DIR)/$(BINARY_NAME)
ENTITLEMENTS = entitlements.plist

# Default signing identity (can be overridden)
SIGNING_IDENTITY ?= 3352WVT944

# Phony targets
.PHONY: all build sign clean help

# Default target
all: build sign

# Build the project in release mode
build:
	@echo "Building $(BINARY_NAME) in release mode..."
	cargo build --release

# Code sign the binary with entitlements
sign: $(BINARY_PATH)
	@echo "Code signing $(BINARY_NAME) with get-task-pid entitlement..."
	codesign --force --sign "$(SIGNING_IDENTITY)" --entitlements $(ENTITLEMENTS) $(BINARY_PATH)
	@echo "Verifying code signature..."
	codesign --verify --verbose $(BINARY_PATH)
	@echo "Checking entitlements..."
	codesign --display --entitlements - $(BINARY_PATH)

# Check if binary exists
$(BINARY_PATH):
	@if [ ! -f $(BINARY_PATH) ]; then \
		echo "Binary not found. Building first..."; \
		$(MAKE) build; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean

# Run the signed binary
run: all
	@echo "Running $(BINARY_NAME)..."
	./$(BINARY_PATH)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build and sign the binary (default)"
	@echo "  build   - Build the project in release mode"
	@echo "  sign    - Code sign the binary with entitlements"
	@echo "  clean   - Clean build artifacts"
	@echo "  run     - Build, sign, and run the binary"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  SIGNING_IDENTITY - Code signing identity (default: -)"
	@echo "    Example: make SIGNING_IDENTITY='Developer ID Application: Your Name'"
