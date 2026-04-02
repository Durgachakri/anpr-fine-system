def calculate_fine(vehicle, speed, no_parking, helmet, seatbelt):
    violations = []
    fine = 0

    # Speed rule
    if speed > 60:
        violations.append("Over Speed")
        fine += 500

    # No parking
    if no_parking == "yes":
        violations.append("No Parking")
        fine += 200

    # Bike rules
    if vehicle.lower() in ["motorcycle", "bike"]:
        if helmet == "no":
            violations.append("No Helmet")
            fine += 1100   # 👈 same as backend

    # Car rules
    else:
        if seatbelt == "no":
            violations.append("No Seatbelt")
            fine += 500

    return violations, fine