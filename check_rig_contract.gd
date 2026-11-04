# Gate a UsdSkel rig layer (.usda) against the rig contract.
#
#   godot --headless -s check_rig_contract.gd -- rig.usda      exit 0 on PASS, 1 on FAIL
#   godot --headless -s check_rig_contract.gd -- --self-test   two positive, four negative controls
#
# The contract comes from the retired browser studio's `[API-Contract] PASS` check, carried
# over to the layer this service writes. Each check measures a position in the layer's own up
# axis and units, read from the file rather than assumed. An unmet precondition is a FAIL,
# never a skip.
#
#   1 skeleton  exactly one Skeleton prim with a non-empty joints array and a restTransforms
#               array of the same length
#   2 names     a hips joint and a head joint are present, by the joint path's last component
#   3 upright   head is above hips along the up axis by at least HEAD_ABOVE_HIPS
#   4 grounded  the lowest joint sits within GROUND_TOLERANCE of 0 (one nickel, 21.2 mm)
#   5 hips      hips sit at least HIPS_MIN_FRACTION of the head height above the ground
#
# The parser reads the usda text this repository emits: `upAxis`, `metersPerUnit`, and inside
# each `def Skeleton` block `uniform token[] joints` and `uniform matrix4d[] restTransforms`
# (row-major, row vectors; world = local x parent). It is not a general USD reader.
extends SceneTree

const GROUND_TOLERANCE := 0.0212
const HEAD_ABOVE_HIPS := 0.10
const HIPS_MIN_FRACTION := 0.30
const HIPS_NAMES := ["hips", "pelvis", "root_hips"]
const HEAD_NAMES := ["head"]


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	var code := 2
	if args.size() == 1 and args[0] == "--self-test":
		code = 0 if self_test() else 1
	elif args.size() == 1:
		code = 0 if report(args[0]) else 1
	else:
		print("usage: godot --headless -s check_rig_contract.gd -- <rig.usda> | --self-test")
	quit(code)


# ---------------------------------------------------------------- usda parsing

func _parse(text: String) -> Dictionary:
	var up := "Y"
	var m := RegEx.create_from_string("upAxis\\s*=\\s*\"([YZ])\"").search(text)
	if m:
		up = m.get_string(1)
	var units := 1.0
	m = RegEx.create_from_string("metersPerUnit\\s*=\\s*([0-9.eE+-]+)").search(text)
	if m:
		units = m.get_string(1).to_float()
	var skels: Array = []
	for d in RegEx.create_from_string("def\\s+Skeleton\\s+\"[^\"]+\"").search_all(text):
		var start := text.find("{", d.get_end())
		var depth := 0
		var i := start
		while i < text.length():
			var c := text[i]
			if c == "{":
				depth += 1
			elif c == "}":
				depth -= 1
				if depth == 0:
					break
			i += 1
		var body := text.substr(start, i - start + 1)
		skels.append(body)
	return {"up": up, "units": units, "skeletons": skels}


func _joints(body: String) -> PackedStringArray:
	var m := RegEx.create_from_string("token\\[\\]\\s+joints\\s*=\\s*\\[([^\\]]*)\\]").search(body)
	var out := PackedStringArray()
	if m == null:
		return out
	for j in RegEx.create_from_string("\"([^\"]*)\"").search_all(m.get_string(1)):
		out.append(j.get_string(1))
	return out


func _rests(body: String) -> Array:
	# matrix4d[] restTransforms = [ ( (a,b,c,d), (..), (..), (..) ), ... ]
	var start := body.find("restTransforms")
	var out: Array = []
	if start < 0:
		return out
	var lb := body.find("[", start)
	var rb := body.find("]", lb)
	var inner := body.substr(lb + 1, rb - lb - 1)
	for mm in RegEx.create_from_string("\\(\\s*(\\([^()]*\\)\\s*,?\\s*){4}\\)").search_all(inner):
		var nums := PackedFloat64Array()
		for n in RegEx.create_from_string("[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?").search_all(mm.get_string()):
			nums.append(n.get_string().to_float())
		if nums.size() == 16:
			out.append(nums)
	return out


func _mul(a: PackedFloat64Array, b: PackedFloat64Array) -> PackedFloat64Array:
	var r := PackedFloat64Array()
	r.resize(16)
	for i in 4:
		for j in 4:
			var s := 0.0
			for k in 4:
				s += a[i * 4 + k] * b[k * 4 + j]
			r[i * 4 + j] = s
	return r


func _world_positions(joints: PackedStringArray, rests: Array, units: float) -> Dictionary:
	# world[j] = rest[j] x world[parent(j)]; the parent is the path with its last component removed
	var world := {}
	var positions := {}
	for idx in joints.size():
		var path := joints[idx]
		var local: PackedFloat64Array = rests[idx]
		var parent := path.get_base_dir()
		var w := local
		if parent != "" and world.has(parent):
			w = _mul(local, world[parent])
		elif parent != "":
			return {}  # a child listed before its parent: the topology is invalid
		world[path] = w
		positions[path] = Vector3(w[12] * units, w[13] * units, w[14] * units)
	return positions


# ---------------------------------------------------------------- the checks

func check(path: String) -> Array:
	var results: Array = []
	if not FileAccess.file_exists(path):
		return [["skeleton", false, "cannot open %s" % path]]
	var doc := _parse(FileAccess.get_file_as_string(path))
	var up_index := 1 if doc["up"] == "Y" else 2
	if doc["skeletons"].size() != 1:
		return [["skeleton", false, "%d skeletons, contract wants 1" % doc["skeletons"].size()]]
	var body: String = doc["skeletons"][0]
	var joints := _joints(body)
	var rests := _rests(body)
	if joints.is_empty():
		return [["skeleton", false, "no joints"]]
	if joints.size() != rests.size():
		return [["skeleton", false, "joints %d != restTransforms %d" % [joints.size(), rests.size()]]]
	var positions := _world_positions(joints, rests, doc["units"])
	if positions.is_empty():
		return [["skeleton", false, "topology: a joint precedes its parent"]]
	results.append(["skeleton", true, "%d joints, up=%s, %s m/unit" % [joints.size(), doc["up"], doc["units"]]])

	var hips := ""
	var head := ""
	for p in positions.keys():
		var leaf: String = p.get_file().to_lower()
		if hips == "" and leaf in HIPS_NAMES:
			hips = p
		if head == "" and leaf in HEAD_NAMES:
			head = p
	var names_ok := hips != "" and head != ""
	results.append(["names", names_ok, "hips=%s head=%s" % [hips, head]])
	if not names_ok:
		return results

	var h_hips: float = positions[hips][up_index]
	var h_head: float = positions[head][up_index]
	var lowest := INF
	for p in positions.values():
		lowest = minf(lowest, p[up_index])
	results.append(["upright", h_head - h_hips >= HEAD_ABOVE_HIPS,
		"head %.3f m, hips %.3f m, head-hips %.3f m (>= %.2f)" % [h_head, h_hips, h_head - h_hips, HEAD_ABOVE_HIPS]])
	results.append(["grounded", absf(lowest) <= GROUND_TOLERANCE,
		"lowest joint %.4f m from ground (|x| <= %.4f, one nickel)" % [lowest, GROUND_TOLERANCE]])
	var height := h_head - lowest
	var frac := (h_hips - lowest) / height if height > 0.0 else 0.0
	results.append(["hips", height > 0.0 and frac >= HIPS_MIN_FRACTION,
		"hips at %.2f of head height (>= %.2f)" % [frac, HIPS_MIN_FRACTION]])
	return results


func report(path: String) -> bool:
	var results := check(path)
	var passed := true
	var n_ok := 0
	for r in results:
		passed = passed and r[1]
		n_ok += 1 if r[1] else 0
		print("%s %-9s %s" % ["PASS" if r[1] else "FAIL", r[0], r[2]])
	print("%s rig contract: %d of %d checks, %s" % ["PASS" if passed else "FAIL", n_ok, results.size(), path])
	return passed


# ---------------------------------------------------------------- self-test

func _write_rig(path: String, heights: Dictionary, up: String, units: float, drop_head: bool) -> void:
	var order := ["hips", "spine", "chest", "neck", "head", "foot_l", "foot_r"]
	if drop_head:
		order.erase("head")
	var joints := PackedStringArray()
	var rests := PackedStringArray()
	for j in order:
		joints.append("\"%s\"" % j)
		var t := [0.0, 0.0, 0.0]
		t[1 if up == "Y" else 2] = heights[j] / units
		rests.append("( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (%s, %s, %s, 1) )" % [t[0], t[1], t[2]])
	var text := "#usda 1.0\n(\n    defaultPrim = \"Asset\"\n    metersPerUnit = %s\n    upAxis = \"%s\"\n)\n\ndef Xform \"Asset\"\n{\n    def \"Rig\"\n    {\n        def Skeleton \"Skeleton\"\n        {\n            uniform token[] joints = [%s]\n            uniform matrix4d[] restTransforms = [%s]\n        }\n    }\n}\n" % [units, up, ", ".join(joints), ", ".join(rests)]
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()


func self_test() -> bool:
	var good := {"hips": 0.95, "spine": 1.10, "chest": 1.30, "neck": 1.55, "head": 1.65, "foot_l": 0.0, "foot_r": 0.0}
	var floating := {}
	for k in good:
		floating[k] = good[k] + 0.5
	var inverted := good.duplicate()
	inverted["hips"] = 1.65
	inverted["head"] = 0.95
	var hips_high := good.duplicate()
	hips_high["hips"] = 1.60
	var cases := [
		["positive upright grounded Y", good, "Y", 1.0, false, true],
		["positive Z-up centimetres", good, "Z", 0.01, false, true],
		["negative inverted", inverted, "Y", 1.0, false, false],
		["negative floating 0.5 m", floating, "Y", 1.0, false, false],
		["negative hips at the head", hips_high, "Y", 1.0, false, false],
		["negative no head joint", good, "Y", 1.0, true, false],
	]
	var dir := OS.get_temp_dir().path_join("rig-contract-%d" % Time.get_ticks_usec())
	DirAccess.make_dir_recursive_absolute(dir)
	var all_ok := true
	var negatives := 0
	for c in cases:
		var p := dir.path_join(String(c[0]).replace(" ", "_") + ".usda")
		_write_rig(p, c[1], c[2], c[3], c[4])
		var got := true
		for r in check(p):
			got = got and r[1]
		var ok: bool = got == c[5]
		all_ok = all_ok and ok
		negatives += 0 if c[5] else 1
		print("%s self-test %s: gate said %s, expected %s" % ["PASS" if ok else "FAIL", c[0], "PASS" if got else "FAIL", "PASS" if c[5] else "FAIL"])
	print("%s self-test: %d cases, %d negative controls" % ["PASS" if all_ok else "FAIL", cases.size(), negatives])
	return all_ok
