# Analisis Akademik & Teknis: `Nodes.py` (DRAMCO LoRa Multi‑Hop Simulator)

> **Batasan analisis:** Seluruh uraian di bawah ini murni didasarkan pada isi kode `Nodes.py`. Rujukan ke paper 2023 *“An Energy‑Efficient LoRa Multi‑Hop Protocol through Preamble Sampling for Remote Sensing”* hanya digunakan untuk menegaskan keselarasan konsep (preamble sampling, forwarding multi‑hop, dan energy‑aware behavior) tanpa menambahkan teori/rumus eksternal.

---

## 🅐 Tujuan File

**Peran fungsional.** `Nodes.py` mendefinisikan model *node* dalam simulasi multi‑hop LoRa, mencakup: status operasi (CAD, RX, TX, PREAMBLE_TX, SLEEP, SENSING), penjadwalan aktivitas dengan **SimPy**, mekanisme **route discovery** & *forwarding*, antrian data (*own data* dan *forwarded data*), deteksi & penanganan **collision**, serta **akuntansi energi dan metrik performa** (PDR/PLR/latency/energy per byte).

**Kait ke layer.**
- **PHY**: representasi *preamble TX*, *CAD* (channel activity detection), serta konsumsi daya per status radio.
- **MAC**: logika bangun‑tidur periodik (periodic wake‑up), *backoff* timer, pengaturan *receive window* via state RX, dan agregasi/pengiriman muatan.
- **Network**: *route discovery*, tabel link (`LinkTable`), pemilihan next hop (`Route`), dan *forwarding* multi‑hop.
- **Energy**: perhitungan energi berbasis waktu‑di‑state × daya‑state.

**Korelasi paper 2023 (konteks aman).** Kode `Nodes.py` mengimplementasikan **preamble sampling** (state `STATE_PREAMBLE_TX` & proses `cad()`/`receiving()`), **multi‑hop forwarding** (objek `Route`, `LinkTable`, *routed message*), dan strategi penghematan energi melalui *duty‑cycling* (SLEEP → CAD → RX/TX) sebagaimana praktik yang dijelaskan dalam karya tersebut—tanpa memperkenalkan teori/rumus baru.

---

## 🅑 Struktur & Enumerasi

**Susunan impor utama**
```python
from .utils import *
from .Timers import TxTimer, TimerType
from .Packets import Message, MessageType
from .Links import LinkTable
from .Routes import Route
import simpy, numpy as np, math, logging, collections
from aenum import Enum, MultiValue, auto, IntEnum
```
**Kelas & fungsi tingkat atas**
- `GatewayState(IntEnum)` – status logis untuk node gateway.
- `NodeType(IntEnum)` – tipe node: `GATEWAY` atau `SENSOR`.
- `NodeState(Enum)` – status operasional node: `STATE_INIT`, `STATE_CAD`, `STATE_RX`, `STATE_TX`, `STATE_PREAMBLE_TX`, `STATE_SLEEP`, `STATE_SENSING` (dengan *fullname* untuk pelabelan).
- `power_of_state(settings, s)` – memetakan `NodeState` → daya (mW) dari `settings`.
- `Node` – model perilaku node dengan ~27 metode (SimPy processes, timers, TX/RX, collision, metrik, plotting).

**Enumerasi & konstanta penting**
- `GatewayState`: `STATE_INIT`, `STATE_ROUTE_DISCOVERY`, `STATE_RX`, `STATE_PROC` — konteks alur gateway.
- `NodeType`: `GATEWAY`, `SENSOR` — menentukan aktivasi timer & perilaku TX khusus gateway (beacon/route discovery).
- `NodeState` (dengan *fullname*): `STATE_INIT`, `STATE_CAD`, `STATE_RX`, `STATE_TX`, `STATE_PREAMBLE_TX`, `STATE_SLEEP`, `STATE_SENSING` — dipakai untuk *state machine* dan akumulasi energi per state.
- `power_of_state(...)` – fungsi *switch‑like* yang mengembalikan **daya per state** dari parameter `settings`.

**Konteks siklus hidup.** Node berputar melalui **SLEEP → CAD → (RX | TX) → SLEEP**, dengan interupsi oleh **SENSING** (pengukuran & penambahan *payload*) dan **ROUTE_DISCOVERY** (khusus gateway atau saat *forwarding*). Status dan waktunya dicatat untuk kalkulasi energi & plotting.

---

## 🅒 `class Node`

### 1) Tujuan Kelas
Memodelkan perilaku *end device/gateway* LoRa multi‑hop di SimPy: membangunkan radio secara periodik (CAD), mendeteksi preamble, menerima/memancarkan paket, mengatur *route discovery* & *forwarding*, menunda TX saat kanal sibuk, serta menghitung metrik energi & kinerja.

### 2) Atribut Utama (ringkas)
Atribut diinisialisasi dalam `__init__` dan diperbarui di `add_meta()`/`timers_setup()`:
- **Lingkungan & konfigurasi:** `env: simpy.Environment`, `settings`.
- **Identitas & posisi:** `uid`, `position`, `type: NodeType`.
- **Status & jejak waktu:** `state`, `states`, `states_time`, `time_spent_in: dict[state→durasi]`.
- **Energi:** `energy_mJ: float` (akumulasi energi), melalui **E += Δt × P(state)** saat `state_change()`.
- **Buffer data:** `data_buffer` (own data), `forwarded_mgs_buffer` (data *forwarded*), `data_created_at` (timestamp muatan pertama).
- **Routing & link:** `route: Route`, `link_table: LinkTable`, `messages_seen: deque(maxlen=MAX_SEEN_PACKETS)`.
- **Statistik trafik:** `messages_sent`, `messages_for_me`, `own_payloads_sent`, `own_payloads_arrived_at_gateway`, `own_payloads_collided`, `forwarded_payloads`, `forwarded_from`.
- **Timer:** `tx_collision_timer`, `tx_route_discovery_timer` (gateway), `tx_aggregation_timer`, `sense_timer` (Semua `TxTimer`).
- **Penunjang TX:** `message_in_tx`, `route_discovery_forward_buffer`.

### 3) Inisialisasi `__init__`
**Cuplikan ringkas**
```python
def __init__(self, env, _settings, _id, _position, _type, **kwargs):
    self.env, self.settings = env, _settings
    self.uid, self.position, self.type = _id, _position, _type
    self.collisions = []
    self.messages_sent, self.messages_for_me = [], []
    self.own_payloads_sent, self.own_payloads_arrived_at_gateway = [], []
    self.own_payloads_collided = []; self.forwarded_payloads = []
    self.data_buffer, self.forwarded_mgs_buffer = [], []
    self.messages_seen = collections.deque(maxlen=self.settings.MAX_SEEN_PACKETS)
    self.state = None; self.states, self.states_time = [], []
    self.energy_mJ = 0; self.time_spent_in = {}
    self.route = Route(); self.link_table = None; self.nodes = []
    self.tx_collision_timer = self.tx_route_discovery_timer = None
    self.tx_aggregation_timer = self.sense_timer = None
    self.timers_setup()
```
**Penjelasan.** Objek menyiapkan struktur log (statistik & buffer), *state machine*, *routing/link context*, serta *timers*. Nilai awal energi 0 mJ; semua durasi per state akan diakumulasi saat *state transition* terjadi.

### 4) Fungsi Utama (per `def`)
> Format per fungsi: **(1) Potongan kode**, **(2) Tujuan**, **(3) Parameter/Return**, **(4) Langkah**, **(5) Operasi penting/rumus**, **(6) Interaksi modul**, **(7) Analisis akademik**, **(8) Konteks paper (opsional)**.

#### a) `add_meta(self, _settings, nodes, link_table)`
1) **Kode**
```python
def add_meta(self, _settings, nodes, link_table):
    self.nodes = nodes
    self.link_table = link_table
    self.settings = _settings
```
2) **Tujuan**: Menyisipkan referensi ke daftar node dan tabel link setelah konstruksi jaringan.
3) **Param/Return**: *void*; memperbarui referensi internal.
4) **Langkah**: Set atribut → selesai.
5) **Rumus**: —
6) **Interaksi**: `LinkTable`, daftar `nodes` untuk query state tetangga.
7) **Analisis**: Pemisahan *constructor* vs *wiring* memudahkan bootstrap topologi.
8) **Konteks**: Menunjang *multi‑hop forwarding* dengan info tetangga.

#### b) `state_change(self, state_to)`
1) **Kode**
```python
def state_change(self, state_to):
    if state_to is not self.state:
        if len(self.states_time) > 0:
            dt = self.env.now - self.states_time[-1]
            self.energy_mJ += dt * power_of_state(self.settings, self.state)
            self.time_spent_in[self.state.name] += dt
        self.state = state_to
        self.states.append(state_to)
        self.states_time.append(self.env.now)
```
2) **Tujuan**: Transisi state + akuntansi energi & durasi per state.
3) **Param/Return**: `state_to: NodeState`; *void*.
4) **Langkah**: Hitung `dt` sejak state terakhir → **E += dt × P(state)** → catat state & timestamp.
5) **Rumus nyata**: `E_total = Σ (Δt × P_state)`; `Δt = now − t_prev`.
6) **Interaksi**: `power_of_state()` → `settings` (daya per state).
7) **Analisis**: Inti model energi granular per state; efisien untuk *post‑hoc* metrik.
8) **Konteks**: Mendukung evaluasi *duty‑cycling* (preamble sampling vs sleep).

#### c) `timers_setup(self)`
1) **Kode**
```python
def timers_setup(self):
    self.tx_collision_timer = TxTimer(self.env, self.settings, TimerType.COLLISION)
    if self.type == NodeType.GATEWAY:
        self.tx_route_discovery_timer = TxTimer(self.env, self.settings, TimerType.ROUTE_DISCOVERY)
    else:
        self.tx_route_discovery_timer = None
    self.tx_aggregation_timer = TxTimer(self.env, self.settings, TimerType.AGGREGATION)
    self.sense_timer = TxTimer(self.env, self.settings, TimerType.SENSE)
```
2) **Tujuan**: Inisialisasi timer untuk *collision backoff*, *route discovery*, *aggregation*, dan *sensing*.
3) **Param/Return**: *void*.
4) **Langkah**: Buat objek `TxTimer` sesuai tipe node.
5) **Rumus**: —
6) **Interaksi**: `TxTimer`, `TimerType`.
7) **Analisis**: Timer memediasi *event scheduling* → kunci efisiensi kanal/energi.
8) **Konteks**: Backoff/aggregation selaras penghematan energi via batch TX.

#### d) `run(self)`
1) **Kode**
```python
def run(self):
    random_wait = np.random.uniform(0, self.settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
    yield self.env.timeout(random_wait)
    self.timers_setup()
    if self.type == NodeType.GATEWAY:
        self.tx_route_discovery_timer.backoff = 1.1 * self.settings.MAX_DELAY_START_PER_NODE_RANDOM_S
        self.tx_route_discovery_timer.start()
    else:
        self.sense_timer.start()
    self.state_change(NodeState.STATE_INIT)
    self.env.process(self.periodic_wakeup())
```
2) **Tujuan**: *Bootstrap* node dan memulai siklus periodik.
3) **Param/Return**: *SimPy process* (generator).
4) **Langkah**: *Random start* → timer → state INIT → jalankan `periodic_wakeup()`.
5) **Rumus**: —
6) **Interaksi**: `TxTimer`, `periodic_wakeup()`.
7) **Analisis**: *Randomized start* mengurangi *synchronization collision* awal.
8) **Konteks**: Sejalan *asynchronous preamble sampling*.

#### e) `check_sensing(self)`
1) **Kode**
```python
def check_sensing(self):
    if (self.sense_until is None and self.sense_timer.is_expired()) or \
       (self.sense_timer.is_expired() and self.env.now < self.sense_until):
        self.state_change(NodeState.STATE_SENSING)
        yield self.env.timeout(self.settings.MEASURE_DURATION_S)
        self.tx_aggregation_timer.start(restart=False)
        self.sense_timer.start()
        if len(self.data_buffer) == 0:
            self.data_created_at = self.env.now
        self.data_buffer.extend(self.application_counter.to_bytes(
            self.settings.MEASURE_PAYLOAD_SIZE_BYTE, 'big'))
        self.application_counter = (self.application_counter + 1) % 65535
```
2) **Tujuan**: Mengukur/sampling data aplikasi dan menambah ke `data_buffer`.
3) **Param/Return**: *SimPy process*; *void*.
4) **Langkah**: Cek timer → masuk `SENSING` → *sleep* `MEASURE_DURATION_S` → start timer agregasi → tambahkan payload.
5) **Rumus**: Inkrement *counter* modul \(\(\mathrm{counter} = (counter + 1) \bmod 65535\)\).
6) **Interaksi**: `tx_aggregation_timer`, `sense_timer`.
7) **Analisis**: Decoupling sensing–TX: memungkinkan **agregasi** sebelum TX.
8) **Konteks**: Agregasi muatan menurunkan biaya *per‑byte* (selaras prinsip hemat energi).

#### f) `check_transmit(self)`
1) **Kode**
```python
def check_transmit(self):
    if self.type == NodeType.GATEWAY and self.tx_route_discovery_timer.is_expired():
        self.route_discovery_forward_buffer = Message(
            self.settings, MessageType.TYPE_ROUTE_DISCOVERY, 0, 0, 0, 0,
            [0x55, 0x55, 0x55], self.env.now, self, [])
        yield self.env.process(self.tx())
        self.tx_route_discovery_timer.backoff = self.settings.ROUTE_DISCOVERY_S
        self.tx_route_discovery_timer.reset()
    elif self.type == NodeType.SENSOR:
        if self.tx_collision_timer.is_expired():
            yield self.env.process(self.tx()); self.tx_collision_timer.reset()
        elif self.tx_aggregation_timer.is_expired() or self.full_buffer():
            yield self.env.process(self.tx()); self.tx_aggregation_timer.reset()
```
2) **Tujuan**: Menentukan kapan node memulai TX (beacon/route discovery atau data/forwarding).
3) **Param/Return**: *SimPy process*; *void*.
4) **Langkah**: Gateway → kirim *route discovery*; Sensor → kirim saat *collision timer* habis (tunda‑TX) atau timer agregasi habis / buffer penuh.
5) **Rumus**: —
6) **Interaksi**: `Message`, `TimerType`, `tx()`.
7) **Analisis**: *Collision timer* menunda TX ketika kanal sibuk; *aggregation timer* mengendalikan batch‑size muatan.
8) **Konteks**: Sesuai gagasan mengurangi airtime & collision.

#### g) `periodic_wakeup(self)`
1) **Kode**
```python
def periodic_wakeup(self):
    while True:
        self.state_change(NodeState.STATE_SLEEP)
        cad_interval = self.settings.CAD_INTERVAL + random(self.settings.CAD_INTERVAL_RANDOM_S)
        yield self.env.timeout(cad_interval)
        cad_detected = yield self.env.process(self.cad())
        if cad_detected:
            packet_for_us = yield self.env.process(self.receiving())
            self.postpone_pending_tx()
        else:
            yield self.env.process(self.check_transmit())
        if self.type != NodeType.GATEWAY:
            yield self.env.process(self.check_sensing())
```
2) **Tujuan**: Siklus *sleep–CAD–(RX|TX)* periodik.
3) **Param/Return**: *SimPy process*.
4) **Langkah**: Tidur → jeda acak → CAD → RX bila deteksi; bila tidak → peluang TX; lalu sensing (non‑gateway).
5) **Rumus**: —
6) **Interaksi**: `cad()`, `receiving()`, `check_transmit()`, `postpone_pending_tx()`.
7) **Analisis**: Pola *duty‑cycling* klasik; randomisasi mengurangi tabrakan terkoordinasi.
8) **Konteks**: Preamble sampling → bangun hanya saat kanal aktif.

#### h) `cad(self)`
1) **Kode**
```python
def cad(self):
    self.state_change(NodeState.STATE_CAD)
    yield self.env.timeout(self.settings.TIME_CAD_WAKE_S + self.settings.TIME_CAD_STABILIZE_S)
    nodes_sending_preamble = self.get_nodes_in_state([NodeState.STATE_PREAMBLE_TX])
    # ... pilih kandidat yang terkuat di atas ambang; hasilkan True/False
```
2) **Tujuan**: *Channel Activity Detection*—mendeteksi preamble TX yang sedang berlangsung.
3) **Param/Return**: *SimPy process*; `bool` (terdeteksi/tidak).
4) **Langkah**: Masuk CAD → *settling time* → list tetangga `STATE_PREAMBLE_TX` → evaluasi kekuatan sinyal terhadap ambang.
5) **Rumus**: Durasi tunggu CAD = `TIME_CAD_WAKE_S + TIME_CAD_STABILIZE_S`.
6) **Interaksi**: `get_nodes_in_state()`, `LinkTable.rss()`/`snr()` (via modul lain saat evaluasi pra‑RX).
7) **Analisis**: Meminimumkan waktu RX aktif dengan *gated listening* berbasis preamble.
8) **Konteks**: Selaras mekanisme *preamble sampling* pada paper.

#### i) `receiving(self)`
1) **Kode**
```python
def receiving(self):
    self.state_change(NodeState.STATE_RX)
    current_active_nodes = self.get_nodes_in_state([STATE_PREAMBLE_TX, STATE_TX])
    while len(current_active_nodes) > 0:
        current_loudest_node = self.check_and_handle_collisions(current_active_nodes)
        # ... logika champion‑based: siapa paling kuat & apakah selesai TX
        # ... proses rx_message untuk kita / forward
```
2) **Tujuan**: Menerima payload saat kanal aktif; *collision handling* berbasis pembanding daya.
3) **Param/Return**: *SimPy process*; `bool packet_for_us`.
4) **Langkah**: Masuk RX → iterasi selama ada TX aktif → tentukan *loudest* → proses selesai TX → `handle_rx_msg()`.
5) **Rumus**: — (seleksi maksimal berdasarkan *rss* dan ambang yang diatur di `settings`).
6) **Interaksi**: `check_and_handle_collisions()`, `handle_rx_msg()`.
7) **Analisis**: *Capture effect* sederhana (node terkuat menang) + pencatatan collision.
8) **Konteks**: Praktik umum kanal LoRa untuk demodulasi pesan yang dominan.

#### j) `tx(self, route_discovery: bool = False)`
1) **Kode**
```python
def tx(self, route_discovery=False):
    if self.route_discovery_forward_buffer is not None:
        route_discovery = True
        self.message_in_tx = self.route_discovery_forward_buffer
    elif len(self.data_buffer) > 0 or len(self.forwarded_mgs_buffer) > 0:
        # Bangun Message(TYPE_ROUTED, hops, lqi kumulatif, address=0 sementara)
        self.message_in_tx = Message(..., MessageType.TYPE_ROUTED, ...,
                                     self.uid, self.data_buffer, self.data_created_at,
                                     self, self.forwarded_mgs_buffer)
        # Tentukan exclude nodes (agar tidak looping) → route = self.route.find_route(exclude)
        # Set address next hop; update hop & cumulative_lqi bila forwarding
    # ... ubah state ke PREAMBLE_TX/TX; catat messages_sent; reset buffer jika own data terkirim
```
2) **Tujuan**: Membuat & mengirim paket (route discovery atau routed data + forwarded data).
3) **Param/Return**: `route_discovery: bool`; *SimPy process*; *void*.
4) **Langkah**: Siapkan `message_in_tx` → (opsional) *find_route* → set `address` → jika *forwarded*, `hop()` & update `cumulative_lqi` → jalankan urutan PREAMBLE_TX → TX → selesai.
5) **Rumus**: Penjumlahan LQI kumulatif: `LQI_total += lqi(link)` (langsung dari kode update kumulatif).
6) **Interaksi**: `Message`, `Route.find_route(...)`, `LinkTable.get_from_uid(...).lqi()`.
7) **Analisis**: *Routing‑aware TX* menggabungkan data sendiri & *forwarded* sehingga menekan overhead.
8) **Konteks**: Sejalan *energy‑aware multi‑hop forwarding*.

#### k) `full_buffer(self)`
1) **Kode**
```python
def full_buffer(self):
    size = sum(fw.payload.size() for fw in self.forwarded_mgs_buffer)
    return size > self.settings.MAX_BUF_SIZE_BYTE * self.settings.MAX_BUF_SIZE_THRESHOLD
```
2) **Tujuan**: Menentukan “buffer penuh” untuk memicu TX dini.
3) **Rumus**: `is_full ⇔ Σ size_fwd > MAX_BUF_SIZE_BYTE × MAX_BUF_SIZE_THRESHOLD`.

#### l) `get_nodes_in_state(self, states)`
1) **Kode**
```python
def get_nodes_in_state(self, states):
    return [n for n in self.nodes if n is not self and n.state in states]
```
2) **Tujuan**: Utilitas query status tetangga.

#### m) `check_and_handle_collisions(self, active_nodes)`
1) **Kode**
```python
def check_and_handle_collisions(self, active_nodes):
    if len(active_nodes) > 1:
        power_threshold = self.settings.LORA_POWER_THRESHOLD
        powers = [(a, self.link_table.get(a, self).rss()) for a in active_nodes]
        powers.sort(key=lambda t: t[1], reverse=True)
        if powers[0][0].state == NodeState.STATE_TX and \
           powers[0][1] >= powers[1][1] + power_threshold:
            active_node = powers[0][0]  # champion menang
        else:
            active_node = None  # collision
            for node in active_nodes:
                if node.state == STATE_TX and node.message_in_tx.is_route_discovery() and not node.message_in_tx.collided:
                    node.message_in_tx.handle_collision(); self.collisions.append(self.env.now)
                elif node.state == STATE_TX and node.message_in_tx.is_routed() and \
                     node.message_in_tx.header.address == self.uid and not node.message_in_tx.collided:
                    node.message_in_tx.handle_collision(); self.collisions.append(self.env.now)
    return active_node
```
2) **Tujuan**: Seleksi *winner* berbasis daya dan tandai collision.
3) **Rumus**: *Capture* sederhana dengan ambang `power_threshold`.
4) **Interaksi**: `LinkTable.rss()`, `Message.handle_collision()`.
5) **Analisis**: Mengaproksimasi *near‑far/capture*; menandai collision spesifik (route discovery & paket untuk node ini).

#### n) `handle_rx_msg(self, rx_packet)`
1) **Kode**
```python
def handle_rx_msg(self, rx_packet):
    if rx_packet.is_route_discovery():
        if self.link_table.get_from_uid(self.uid, rx_packet.header.address).in_range():
            self.route.update(rx_packet.header.address,
                              self.link_table.get_from_uid(self.uid, rx_packet.header.address).snr(),
                              rx_packet.header.cumulative_lqi + self.link_table.get_from_uid(
                                  self.uid, rx_packet.header.address).lqi(),
                              rx_packet.header.hops)
    if rx_packet.header.uid in self.messages_seen: return False
    if rx_packet.is_route_discovery():
        self.route_discovery_forward_buffer = rx_packet.copy()
        # ... set alamat tujuan baru untuk forwarding
    elif rx_packet.is_routed():
        # ... jika alamatnya kita → proses; else → siapkan untuk forward
```
2) **Tujuan**: Memperbarui rute dari beacon & memutuskan *consume/forward* paket routed.
3) **Rumus**: `LQI_kumulatif_baru = LQI_rx + LQI_link(uid→addr)`.
4) **Interaksi**: `Route.update(...)`, `LinkTable.get_from_uid(...).{snr,lqi}()`.
5) **Analisis**: *Learning‑based routing* via pesan route discovery; mencegah re‑proses via `messages_seen`.

#### o) `arrived_at_gateway(self, payload)`
1) **Kode**
```python
def arrived_at_gateway(self, payload):
    if payload.size()-3 > settings.MEASURE_PAYLOAD_SIZE_BYTE:
        for i in range(0, math.floor((payload.size()-3)/settings.MEASURE_PAYLOAD_SIZE_BYTE)):
            cp = payload.copy(); cp.clip(i)
            self.own_payloads_arrived_at_gateway.append(cp)
    else:
        self.own_payloads_arrived_at_gateway.append(payload)
```
2) **Tujuan**: Normalisasi muatan yang sampai (split/clip) dan pencatatan kedatangan.
3) **Rumus**: Perhitungan jumlah *chunks* = `⌊(size−3)/MEASURE_PAYLOAD_SIZE_BYTE⌋`.

#### p) `pdr(self)` & `plr(self)`
1) **Kode**
```python
def pdr(self):
    return len(self.own_payloads_arrived_at_gateway) / len(self.own_payloads_sent) if len(self.own_payloads_sent)>0 else 1

def plr(self):
    return len(self.own_payloads_collided) / len(self.own_payloads_sent) if len(self.own_payloads_sent)>0 else 0
```
2) **Rumus**: `PDR = N_arrived / N_sent`; `PLR = N_collided / N_sent`.

#### q) `aggregation_efficiency(self)`
1) **Kode**
```python
def aggregation_efficiency(self):
    if len(self.messages_sent) == 0: return 1
    only_own = sum(1 for m in self.messages_sent if len(m.payload.forwarded_data)==0)
    return 1 - only_own/len(self.messages_sent)
```
2) **Rumus**: `η_agg = 1 − (#msg_only_own / #msg_total)`.

#### r) Energi & Metrik Lain
- `energy(self)` → nilai **E_total (mJ)** tersimpan.
- `energy_per_byte(self)` → `E_total / (Σ bytes_own + Σ bytes_forwarded)`.
- `energy_tx_per_byte(self)` → hanya dari durasi `STATE_PREAMBLE_TX` & `STATE_TX` (energi TX murni) dibagi total byte yang dikirim.
- `energy_per_state(self)` → kamus `{state: durasi(state) × P_state}`.
- `latency(self)` → daftar `payload.arrived_at − payload.created_at`.
- `plot_states(self, axis=None)` → menggambar kurva state vs waktu dan menandai `self.collisions`.

---

## 5) Integrasi Antar‑Modul (yang **digunakan langsung**)
- **`Timers.TxTimer`, `TimerType`**: pengatur *backoff*, agregasi, sensing, dan route discovery.
- **`Packets.Message`, `MessageType`**: menyusun paket *routed*/*route discovery*, *UID*, *hops*, *cumulative_lqi*, *payload split/clip*.
- **`Links.LinkTable`**: menyediakan metrik tautan (mis. `rss()`, `snr()`, `lqi()`, `in_range()`), dipakai untuk *capture/collision* dan update rute.
- **`Routes.Route`**: `update(...)`, `find_route(exclude)` dan `set_fixed(...)` untuk pemilihan next hop.
- **`utils`**: randomisasi kecil (mis. `random(...)`) dan utilitas lain.
- **`simpy`**: *event scheduling* (`yield env.timeout(...)`, `env.process(...)`).

---

## ⚙️ Dinamika & Dampak Komputasional
- **Efisiensi Energi**: Diwujudkan melalui *duty‑cycling* (SLEEP), **CAD** singkat, dan **agregasi** muatan. Rumus energi **eksplisit di kode** adalah:
  - \(E_{total} = \sum (\Delta t_{state} \times P_{state})\)
  - \(E_{/byte} = E_{total} / (B_{own}+B_{fwd})\)
  - \(E_{TX/byte} = (t_{PREAMBLE\_TX} + t_{TX})\times P_{TX} \; / \; (B_{own}+B_{fwd})\)
- **Latency/Throughput**: Dipengaruhi oleh *aggregation timer*, *collision backoff*, dan seleksi *loudest winner*.
- **Collision Handling**: *Capture‑threshold* sederhana berbasis `LORA_POWER_THRESHOLD`; tabrakan menandai paket (`handle_collision()`) dan dicatat di `self.collisions`.
- **Scheduling**: Semua proses (`run`, `periodic_wakeup`, `cad`, `receiving`, `tx`, `check_*`) adalah **generator SimPy**, memastikan determinisme *event‑driven*.

---

## 🧠 Catatan Pemeriksaan Tambahan
- Fungsi yang tercakup: `__init__`, `add_meta`, `state_change`, `timers_setup`, `run`, `check_sensing`, `check_transmit`, `postpone_pending_tx`, `periodic_wakeup`, `receiving`, `tx`, `cad`, `full_buffer`, `get_nodes_in_state`, `check_and_handle_collisions`, `collided`, `handle_rx_msg`, `arrived_at_gateway`, `pdr`, `plr`, `aggregation_efficiency`, `energy`, `energy_per_byte`, `energy_tx_per_byte`, `energy_per_state`, `latency`, `plot_states`.
- Jika ada fungsi lain di modul lain (mis. detail `LinkTable`, `Route`, `TxTimer`, `Message`), **tidak dianalisis di sini** karena di luar `Nodes.py`.

---

## 🎯 Kesimpulan Pembimbing
`Nodes.py` merangkum *state machine* node LoRa multi‑hop yang **energi‑sadar** dengan
(1) akuntansi energi eksplisit per state,
(2) *asynchronous preamble sampling* (CAD→RX),
(3) *routing & forwarding* yang menggabungkan *own* + *forwarded payloads*,
(4) *collision handling* berbasis ambang daya, dan
(5) metrik keluaran (PDR/PLR/latency/energy) yang langsung bersumber dari struktur data internal.

Struktur ini **merefleksikan praktik hemat energi**: *sleep‑first*, *gated listening*, *aggregation before TX*, dan *forwarding aware routing*, tanpa menambahkan teori di luar yang ditulis dalam kode.

---

### 📥 Ekspor (DOCX / PDF / MD)
Gunakan tombol **Export** di panel Canvas ini untuk mengunduh versi:
- **.docx** — formula linear (mis. `E_total = Σ(Δt × P_state)`) kompatibel **Microsoft Word Equation**.
- **.pdf** — untuk *camera‑ready* review.
- **.md** — untuk versi *lightweight* (Git/Colab/README).

> Jika Anda ingin saya membagi dokumen ini menjadi *bab per fungsi* atau menambahkan tabel ringkas atribut/timer, beri tahu—saya siapkan varian siap ekspor yang lebih terstruktur.

