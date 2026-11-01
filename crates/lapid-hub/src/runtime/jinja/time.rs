//! `strftime_now(fmt)` for chat templates (transformers passes `datetime.now().strftime`).
//! UTC clock; `LAPID_NOW=<unix seconds>` pins it (reproducible renders in verification).

const MONTHS: [&str; 12] =
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

fn now_secs() -> i64 {
    if let Some(s) = std::env::var("LAPID_NOW").ok().and_then(|v| v.trim().parse::<i64>().ok()) {
        return s;
    }
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// (year, month 1..=12, day 1..=31) of days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Python `strftime` subset: %Y %y %m %d %e %b %B %a %A %H %I %M %S %p %j %%.
pub(crate) fn strftime(fmt: &str, secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil(days);
    let (hh, mm, ss) = (tod / 3600, (tod / 60) % 60, tod % 60);
    let wd = (days + 3).rem_euclid(7) as usize; // 1970-01-01 was a Thursday
    let cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let yday = cum[(m - 1) as usize] + d + u32::from(m > 2 && is_leap(y));
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('b') => out.push_str(&MONTHS[(m - 1) as usize][..3]),
            Some('B') => out.push_str(MONTHS[(m - 1) as usize]),
            Some('a') => out.push_str(&DAYS[wd][..3]),
            Some('A') => out.push_str(DAYS[wd]),
            Some('H') => out.push_str(&format!("{hh:02}")),
            Some('I') => out.push_str(&format!("{:02}", if hh % 12 == 0 { 12 } else { hh % 12 })),
            Some('M') => out.push_str(&format!("{mm:02}")),
            Some('S') => out.push_str(&format!("{ss:02}")),
            Some('p') => out.push_str(if hh < 12 { "AM" } else { "PM" }),
            Some('j') => out.push_str(&format!("{yday:03}")),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

pub(crate) fn strftime_now(fmt: &str) -> String {
    strftime(fmt, now_secs())
}

#[cfg(test)]
mod tests {
    use super::strftime;

    #[test]
    fn dates() {
        // 2024-07-26 13:25:09 UTC (a Friday)
        assert_eq!(strftime("%d %b %Y", 1_722_000_309), "26 Jul 2024");
        assert_eq!(strftime("%A %B %e %H:%M:%S %p %j %y %%", 1_722_000_309), "Friday July 26 13:25:09 PM 208 24 %");
        assert_eq!(strftime("%Y-%m-%d", 0), "1970-01-01");
        assert_eq!(strftime("%Y-%m-%d %a", 951_782_400), "2000-02-29 Tue");
    }
}
