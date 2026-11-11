extends Node3D
## 사전 정의된 합성 명령을 객체와 분리된 Scene API로 적용한다.

var subject: MeshInstance3D
var report: FileAccess
var commands: Array = []
var cursor := 0
var elapsed := 0.0
var test_mode := false
var frame_count := 0
var applied := {}
var active_action := "STOP"
var active_intent := ""
var live_port := 0
var peer: StreamPeerTCP
var wire := PackedByteArray()
var last_live_msec := 0
var live_connected := false
const ACTIONS := ["STOP", "ROTATE_CW", "ROTATE_CCW", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"]

func _ready() -> void:
    var args := OS.get_cmdline_user_args()
    var command_path := ""
    var report_path := ""
    var object_kind := "box"
    for i in range(args.size()):
        if args[i] == "--test":
            test_mode = true
        elif args[i] == "--commands" and i + 1 < args.size():
            command_path = args[i + 1]
        elif args[i] == "--report" and i + 1 < args.size():
            report_path = args[i + 1]
        elif args[i] == "--port" and i + 1 < args.size():
            live_port = int(args[i + 1])
        elif args[i] == "--object" and i + 1 < args.size():
            object_kind = args[i + 1]
    subject = MeshInstance3D.new()
    subject.mesh = SphereMesh.new() if object_kind == "sphere" else BoxMesh.new()
    add_child(subject)
    var material := StandardMaterial3D.new()
    material.albedo_color = Color(0.2, 0.8, 0.7)
    subject.material_override = material
    var camera := Camera3D.new()
    camera.position = Vector3(0, 1.5, 5)
    add_child(camera)
    camera.look_at(Vector3.ZERO)
    var light := DirectionalLight3D.new()
    light.rotation_degrees = Vector3(-40, -30, 0)
    add_child(light)
    var label := Label.new()
    label.text = "합성 명령 재생 · 실제 EEG 해독 및 NPU 검증 아님"
    label.position = Vector2(16, 16)
    add_child(label)
    if (live_port == 0 and not FileAccess.file_exists(command_path)) or report_path.is_empty():
        push_error("명령 파일과 새 보고서 경로를 지정하세요.")
        get_tree().quit(1)
        return
    if FileAccess.file_exists(report_path):
        push_error("기존 보고서는 덮어쓰지 않습니다.")
        get_tree().quit(1)
        return
    if live_port == 0:
        var source := FileAccess.open(command_path, FileAccess.READ)
        while source.get_position() < source.get_length():
            var line := source.get_line()
            if line.strip_edges().is_empty():
                continue
            var command = JSON.parse_string(line)
            if not command is Dictionary:
                push_error("장면 명령 JSON 형식이 잘못되었습니다.")
                get_tree().quit(1)
                return
            commands.append(command)
    report = FileAccess.open(report_path, FileAccess.WRITE)
    if report == null:
        push_error("렌더 보고서를 기록할 수 없습니다.")
        get_tree().quit(1)
        return
    if live_port > 0:
        peer = StreamPeerTCP.new()
        last_live_msec = Time.get_ticks_msec()
        if peer.connect_to_host("127.0.0.1", live_port) != OK:
            finish_live("TCP_CONNECT_FAILED")

func apply_command(command: Dictionary, delta: float) -> bool:
    var action = command.get("action")
    var intent = command.get("source_intent_id")
    var magnitude = command.get("magnitude")
    var duration = command.get("duration_ms")
    if (action not in ACTIONS or not intent is String or intent.is_empty()
            or applied.has(intent) or command.get("validation_status") != "VALIDATED_SYNTHETIC"
            or command.get("render_policy_version") != "synthetic-v1"
            or command.get("scene_id") != "demo" or command.get("object_id") != "subject"
            or not (magnitude is float or magnitude is int) or not is_finite(magnitude)
            or magnitude < 0 or magnitude > 1
            or not (duration is float or duration is int) or not is_finite(duration)
            or duration <= 0 or duration > 1000):
        active_action = "STOP"
        return false
    delta = duration / 1000.0
    applied[intent] = true
    active_action = action
    active_intent = intent
    match action:
        "ROTATE_CW": subject.rotate_y(-magnitude * delta)
        "ROTATE_CCW": subject.rotate_y(magnitude * delta)
        "MOVE_LEFT": subject.position.x -= magnitude * delta
        "MOVE_RIGHT": subject.position.x += magnitude * delta
        "MOVE_UP": subject.position.y += magnitude * delta
        "MOVE_DOWN": subject.position.y -= magnitude * delta
    return true

func _process(delta: float) -> void:
    frame_count += 1
    if report == null:
        return
    if live_port > 0:
        process_live()
        return
    elapsed += delta
    if cursor >= commands.size():
        active_action = "STOP"
        report.flush()
        if test_mode:
            report.close()
            get_tree().quit(0)
        return
    if not test_mode and elapsed < 0.05:
        return
    elapsed = 0
    var command: Dictionary = commands[cursor]
    cursor += 1
    record_command(command)

func record_command(command: Dictionary) -> void:
    var success := apply_command(command, 0.05)
    var row := {
        "render_id": "render-%d" % cursor,
        "source_intent_id": command.get("source_intent_id"),
        "pc_applied_mono_us": Time.get_ticks_usec(),
        "frame_id": frame_count,
        "action": active_action,
        "accepted": success,
        "actual_object_transform": {
            "position": [subject.position.x, subject.position.y, subject.position.z],
            "rotation": [subject.rotation.x, subject.rotation.y, subject.rotation.z]
        },
        "fps": Engine.get_frames_per_second(),
        "source": "SYNTHETIC_SCRIPTED_REPLAY"
    }
    report.store_line(JSON.stringify(row))
    report.flush()

func finish_live(reason: String) -> void:
    active_action = "STOP"
    if report != null:
        report.store_line(JSON.stringify({
            "action": "STOP", "reason": reason, "source": "SYNTHETIC_LIVE_CONTROL",
            "frame_id": frame_count, "pc_applied_mono_us": Time.get_ticks_usec()
        }))
        report.flush()
    if peer != null:
        peer.disconnect_from_host()
    if test_mode:
        if report != null:
            report.close()
        get_tree().quit(0 if reason == "TCP_EOF" else 1)
    live_port = 0
    commands.clear()

func process_live() -> void:
    peer.poll()
    var status := peer.get_status()
    if status == StreamPeerTCP.STATUS_CONNECTED:
        live_connected = true
        var available := peer.get_available_bytes()
        if available > 0:
            if available > 65536:
                finish_live("ERROR_SCHEMA")
                return
            var received := peer.get_data(available)
            if received[0] != OK:
                finish_live("TCP_READ_FAILED")
                return
            wire.append_array(received[1])
        while wire.size() >= 4:
            var length := (int(wire[0]) << 24) | (int(wire[1]) << 16) | (int(wire[2]) << 8) | int(wire[3])
            if length <= 0 or length > 16384:
                finish_live("ERROR_SCHEMA")
                return
            if wire.size() < length + 4:
                break
            var command = JSON.parse_string(wire.slice(4, length + 4).get_string_from_utf8())
            wire = wire.slice(length + 4)
            if not command is Dictionary:
                finish_live("ERROR_SCHEMA")
                return
            last_live_msec = Time.get_ticks_msec()
            cursor += 1
            record_command(command)
            if active_action == "STOP" and not applied.has(command.get("source_intent_id")):
                finish_live("ERROR_SCHEMA")
                return
    elif status == StreamPeerTCP.STATUS_ERROR or (live_connected and status == StreamPeerTCP.STATUS_NONE):
        finish_live("TCP_EOF" if wire.is_empty() else "TRUNCATED_FRAME")
        return
    if Time.get_ticks_msec() - last_live_msec > 1000:
        finish_live("TIMEOUT")
