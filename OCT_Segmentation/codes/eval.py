def eval(
    val_loader,
    criterion,
    model,
    n_classes,
    dice_s=True,
    device="cuda",
    im_save=False,
    use_morph: bool | None = None,
):

    model.eval()

    counter = 0
    dice = 0
    mae = 0
    total_loss = 0
    dice_all = np.zeros(n_classes)
    per_layer_all = np.zeros(n_classes)

    with torch.no_grad():
        for img, label in tqdm.tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            label = label.squeeze(1)
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

            if use_morph is None:
                pred = model(img)
            else:
                pred = model(img, use_morph=use_morph)

            loss, pred = compute_seg_loss(pred, label, criterion, device)
            total_loss += loss.item()

            max_val, idx = torch.max(pred, 1)
            pred_seg = idx.cpu().data.numpy()
            label_seg = label.cpu().numpy()
            ret = compute_dice(label_seg, pred_seg)
            pa = compute_pa(label_seg, pred_seg)

            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

            if dice_s:
                d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
                dice += d1
                dice_all += d2

            label_mae = torch.nn.functional.one_hot(
                label, num_classes=n_classes
            ).squeeze()
            label_mae = label_mae.permute(0, 3, 1, 2)

            init_mae, per_layer = MAE_New(
                label_mae, pred, n_classes=n_classes, classes=list(range(1, 8))
            )
            mae += init_mae.item()
            per_layer_all += per_layer
            counter += 1

        loss = total_loss / counter
        dice_all = dice_all / counter
        per_layer_all = per_layer_all / counter
        mae = mae / counter  # this is already 7-layer MAE (thanks to classes=[1..7])

        retinal_mean_dice = dice_all[1:8].mean()  # 7-layer Dice
        non_bg_mean_dice = dice_all[1:].mean()  # 8-layer (if including fluid)
        mean_dice_all = dice_all.mean()

        print(
            "Validation loss:",
            loss,
            "\n  Mean Dice (all classes):",
            mean_dice_all,
            "\n  Mean Dice (layers 1–7 only):",
            retinal_mean_dice,
            "\n  Mean Dice (classes 1–8, no bg):",
            non_bg_mean_dice,
            "\n  Dice All:",
            dice_all,
            "\n  MAE (layers 1–7):",
            mae,
            "\n  per layer MAE (1–7):",
            per_layer_all[1:8],
        )

        # return retinal_mean_dice as the main scalar Dice
        return retinal_mean_dice, loss, dice_all, mae, per_layer_all
