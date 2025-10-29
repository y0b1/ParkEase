    if spot_statuses[idx] in (STATUS_OCCUPIED, STATUS_RESERVED):
                spot_statuses[idx] = STATUS_VACANT
                messagebox.showinfo("Thank You", "Thank you!")
            else:
            # It was already vacant (no reservation, no car)
                messagebox.showinfo("Thank You", "Thank you!")
