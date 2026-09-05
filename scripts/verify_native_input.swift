// Uses macOS Accessibility permission to exercise the actual native input path.
// Usage: swift scripts/verify_native_input.swift PID OUTPUT.json
import AppKit
import CoreGraphics
import Foundation
import ApplicationServices
let pid = pid_t(CommandLine.arguments[1])!
let output = CommandLine.arguments[2]
guard let app = NSRunningApplication(processIdentifier: pid) else { fatalError("Viewer PID missing") }
app.activate(options: [])
Thread.sleep(forTimeInterval: 0.5)
let windows = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
guard let window = windows.first(where: { ($0[kCGWindowOwnerPID as String] as? Int) == Int(pid) }),
      let bounds = window[kCGWindowBounds as String] as? [String: CGFloat] else { fatalError("Viewer window missing") }
let x = bounds["X"]! + bounds["Width"]! * 0.65
let y = bounds["Y"]! + bounds["Height"]! * 0.55
var actions = [[String: Any]]()
func foreground() {
    app.activate(options: [])
    let ax = AXUIElementCreateApplication(pid)
    AXUIElementSetAttributeValue(ax, kAXFrontmostAttribute as CFString, kCFBooleanTrue)
    var windows: CFTypeRef?
    if AXUIElementCopyAttributeValue(ax, kAXWindowsAttribute as CFString, &windows) == .success,
       let list = windows as? [AXUIElement], let window = list.first {
        AXUIElementPerformAction(window, kAXRaiseAction as CFString)
    }
}
func mouse(_ type: CGEventType, _ point: CGPoint, _ button: CGMouseButton, _ flags: CGEventFlags = []) {
    let event = CGEvent(mouseEventSource: CGEventSource(stateID: .hidSystemState), mouseType: type, mouseCursorPosition: point, mouseButton: button)!
    event.flags = flags
    event.setIntegerValueField(.mouseEventClickState, value: 1)
    event.post(tap: .cghidEventTap)
}
func key(_ code: CGKeyCode, _ down: Bool) {
    CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: down)!.postToPid(pid)
}
func action(_ name: String, _ body: () -> Void) {
    foreground()
    Thread.sleep(forTimeInterval: 0.25)
    let start = Date().timeIntervalSince1970
    body()
    let end = Date().timeIntervalSince1970
    actions.append(["name": name, "start": start, "end": end])
    Thread.sleep(forTimeInterval: 0.18)
}
let center = CGPoint(x: x, y: y)
mouse(.mouseMoved, center, .left)
mouse(.leftMouseDown, center, .left); mouse(.leftMouseUp, center, .left)
action("left_drag_pan") {
    mouse(.leftMouseDown, center, .left)
    for i in 1...12 {
        mouse(.leftMouseDragged, CGPoint(x: x + CGFloat(i) * 5, y: y + CGFloat(i) * 2), .left)
        Thread.sleep(forTimeInterval: 0.015)
    }
    mouse(.leftMouseUp, CGPoint(x: x + 60, y: y + 24), .left)
}
for (name, code): (String, CGKeyCode) in [("w",13),("s",1),("a",0),("d",2),("arrow_up",126),("arrow_down",125),("arrow_left",123),("arrow_right",124),("q",12),("e",14),("r",15),("f",3)] {
    action(name) { key(code, true); Thread.sleep(forTimeInterval: 0.5); key(code, false) }
}
action("right_drag_orbit") {
    mouse(.mouseMoved, center, .right); mouse(.rightMouseDown, center, .right)
    for i in 1...12 {
        mouse(.rightMouseDragged, CGPoint(x: x + CGFloat(i) * 4, y: y + CGFloat(i)), .right)
        Thread.sleep(forTimeInterval: 0.015)
    }
    mouse(.rightMouseUp, CGPoint(x: x + 48, y: y + 12), .right)
}
for (name, flags): (String, CGEventFlags) in [("cmd_drag_orbit", .maskCommand), ("ctrl_drag_orbit", .maskControl)] {
    action(name) {
        mouse(.mouseMoved, center, .left, flags)
        mouse(.leftMouseDown, center, .left, flags)
        for i in 1...12 {
            mouse(.leftMouseDragged, CGPoint(x: x + CGFloat(i) * 4, y: y + CGFloat(i)), .left, flags)
            Thread.sleep(forTimeInterval: 0.015)
        }
        mouse(.leftMouseUp, CGPoint(x: x + 48, y: y + 12), .left, flags)
        mouse(.mouseMoved, center, .left)
    }
}
action("wheel_zoom") {
    CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: 3, wheel2: 0, wheel3: 0)!.post(tap: .cghidEventTap)
    Thread.sleep(forTimeInterval: 0.3)
}
key(111, true); key(111, false) // F12: capture directly from the app framebuffer.
Thread.sleep(forTimeInterval: 0.7)
let data = try JSONSerialization.data(withJSONObject: actions, options: [.prettyPrinted, .sortedKeys])
try data.write(to: URL(fileURLWithPath: output))
key(53, true); key(53, false) // Escape: close this viewer and flush the report.
print("Exercised \(actions.count) native input actions in PID \(pid)")
