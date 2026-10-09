"""Versioned literature anchors and explicitly approximate physical cost models.

These are reference estimates, not a memory compiler, PDK or timing signoff.
The Table I SRAM energy anchors do not specify word width/ports. Interpolation
and banking overhead are our assumptions; Fig.10 is not an exact macro table.
"""
from functools import lru_cache
import math

NN_BATON = 'https://www.zhanhongtan.com/publication/isca21/isca21.pdf'
GRS = 'https://research.nvidia.com/sites/default/files/pubs/2019-01_A-1.17-pJ/b%2C-25-Gb/s/pin/JSSCC_2019_GRS_FINAL.pdf'


@lru_cache(maxsize=2048)
def sram_bank(spec, name):
    capacities = {'WL1': spec.l1_weight_bytes, 'AL1': spec.l1_activation_bytes,
                  'OL1': spec.l1_output_bytes, 'WL2': spec.l2_weight_bytes,
                  'AL2': spec.l2_activation_bytes, 'OL2': spec.l2_output_bytes,
                  'UB': spec.unified_buffer_bytes}
    capacity = capacities[name]
    bandwidth = (spec.l1_bandwidth_Bps if name.endswith('1') else
                 spec.l2_bandwidth_Bps if name.endswith('2') else spec.unified_bandwidth_Bps)
    if spec.cost_profile == 'legacy_mixed_v0':
        return dict(name=name, capacity_bytes=capacity, banks=1, bytes_per_bank=capacity,
                    allocated_bytes=capacity, word_bits=spec.sram_word_bits,
                    rw_ports=spec.sram_rw_ports, area_mm2=capacity*spec.sram_area_mm2_per_byte,
                    read_pj_per_bit=spec.sram_energy_pj_per_bit,
                    write_pj_per_bit=spec.sram_energy_pj_per_bit,
                    status='historical mixed flat-cost model')
    # Enough independent 1RW banks to provision the requested aggregate service.
    # Scheduler serializes R/W at the hierarchy resource; banking does not grant
    # free bandwidth. Smaller physical macros have their own peripheral cost.
    word_bytes = spec.sram_word_bits / 8
    banks = max(math.ceil(capacity/spec.sram_max_bank_bytes),
                math.ceil(bandwidth/(word_bytes*spec.compute_clock_hz*spec.sram_rw_ports)))
    bank_bytes = math.ceil(capacity/banks/word_bytes)*int(word_bytes)
    allocated = banks*bank_bytes
    # Table I anchors 1KiB .30 and 32KiB .81; log-size interpolation is explicitly
    # a sensitivity-testable approximation, including extrapolation below 1KiB.
    slope = (spec.sram_energy_pj_per_bit-spec.sram_l1_anchor_pj_per_bit)/5
    anchor = max(.01, spec.sram_l1_anchor_pj_per_bit+slope*math.log2(bank_bytes/1024))
    width_factor = (spec.sram_word_bits/128)**spec.sram_width_energy_exponent
    port_factor = 1+spec.sram_extra_port_energy_factor*(spec.sram_rw_ports-1)
    read = anchor*width_factor*port_factor*spec.sram_energy_scale
    write = read*spec.sram_write_energy_factor
    # ~0.55 um2/byte + ~1000 um2 per bank: rough Fig.10 area trend, not exact
    # digitization. Width and extra-port peripheral multipliers are assumptions.
    periphery = spec.sram_bank_overhead_mm2*(spec.sram_word_bits/128)*spec.sram_rw_ports
    area = (allocated*spec.sram_area_mm2_per_byte+banks*periphery)*spec.sram_area_scale
    return dict(name=name, capacity_bytes=capacity, banks=banks, bytes_per_bank=bank_bytes,
                allocated_bytes=allocated, word_bits=spec.sram_word_bits,
                rw_ports=spec.sram_rw_ports, area_mm2=area,
                read_pj_per_bit=read, write_pj_per_bit=write,
                provisioned_bandwidth_Bps=banks*word_bytes*spec.compute_clock_hz*spec.sram_rw_ports,
                status='anchor-interpolated bank estimate; uncalibrated width/ports/area/RW assumptions',
                source=NN_BATON)


def sram_access_energy(spec, ledger):
    """Named reads/writes in bits, returning joules and auditable bank costs."""
    result = {}
    for name, accesses in ledger.items():
        bank = sram_bank(spec, name)
        read, write = accesses.get('read_bits', 0), accesses.get('write_bits', 0)
        if read < 0 or write < 0:
            raise ValueError('SRAM access counts must be nonnegative')
        result[name] = {**accesses, 'energy_j': (read*bank['read_pj_per_bit']+
                       write*bank['write_pj_per_bit'])*1e-12, 'cost': bank}
    return sum(row['energy_j'] for row in result.values()), result


def grs_port(spec, rate_Bps=None):
    """Pay whole endpoint macros: separate TX and RX bricks, each 8 data + clock.

    Conservative full-duplex adaptation, not a claim that the paper implemented
    this exact port. Footprint includes active circuitry and the 20-bump array.
    """
    rate = spec.nop_bandwidth_Bps if rate_Bps is None else rate_Bps
    macros = math.ceil(rate*8/(spec.grs_data_lanes*spec.grs_lane_rate_bps))
    total = 2*macros
    return dict(macros_per_direction=macros, endpoint_macros=total,
                data_lanes=total*spec.grs_data_lanes, clock_lanes=total,
                power_ground_bumps=total*11, total_bumps=total*20,
                per_direction_capacity_Bps=macros*spec.grs_data_lanes*spec.grs_lane_rate_bps/8,
                footprint_mm2=total*spec.grs_phy_area_mm2,
                active_circuitry_mm2=total*spec.grs_active_area_mm2,
                requested_per_direction_Bps=rate, source=GRS,
                policy='one whole 8-data-lane brick per direction per endpoint; active area already inside footprint',
                geometry_status='reference bump-array fit only; package/timing/electrical signoff unavailable')
